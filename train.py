# %%
# # !pip install --upgrade pip
# !pip install --upgrade 
# # !pip install ipywidgets

# %%
# from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict, concatenate_datasets, Audio
from transformers import EvalPrediction, pipeline, AutoModel, AutoTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperTokenizer, WhisperTokenizerFast, WhisperProcessor,WhisperFeatureExtractor
import evaluate
from torch.utils.data import DataLoader
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import time

# torch.set_num_threads(1)

# %%
cache_dir = '/mnt/8THDD0/storage/huggingface'
model_name = "openai/whisper-small"
target_lang = 'zh'
common_voice_11_lang = "zh-HK"
common_voice_17_lang = "yue"
# model_name = "openai/whisper-large-v3-turbo"
output_dir = "/mnt/8THDD0/storage/train/output/"+ model_name.split('/')[1] +"-" + target_lang
model_path = model_name

# %%
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language=target_lang, task="transcribe", cache_dir=cache_dir)
processor = WhisperProcessor.from_pretrained(model_name, language=target_lang, task="transcribe", cache_dir=cache_dir)
# %%
common_voice_11 = DatasetDict()
common_voice_11_1 = DatasetDict()
common_voice_17 = DatasetDict()
common_voice = DatasetDict()
# %%
common_voice_11["train"] = load_dataset("mozilla-foundation/common_voice_11_0", common_voice_11_lang, split="train+validation", trust_remote_code=True, cache_dir=cache_dir)
common_voice_11["test"] = load_dataset("mozilla-foundation/common_voice_11_0", common_voice_11_lang, split="test", trust_remote_code=True, cache_dir=cache_dir)
common_voice_11_1["train"] = load_dataset("mozilla-foundation/common_voice_11_0", common_voice_17_lang, split="train+validation", trust_remote_code=True, cache_dir=cache_dir)
common_voice_11_1["test"] = load_dataset("mozilla-foundation/common_voice_11_0", common_voice_17_lang, split="test", trust_remote_code=True, cache_dir=cache_dir)
common_voice_17["train"] = load_dataset("mozilla-foundation/common_voice_17_0", common_voice_17_lang, split="train+validation", trust_remote_code=True, cache_dir=cache_dir)
common_voice_17["test"] = load_dataset("mozilla-foundation/common_voice_17_0", common_voice_17_lang, split="test", trust_remote_code=True, cache_dir=cache_dir)
common_voice["train"] = concatenate_datasets([common_voice_11["train"], common_voice_11_1["train"], common_voice_17["train"]])
common_voice["test"] = concatenate_datasets([common_voice_11["test"], common_voice_11_1["test"], common_voice_17["test"]])
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print('common train_records: ' + str(len(common_voice["train"])))
print('common test_records: ' + str(len(common_voice["test"])))

# %%
print(common_voice["train"][0])

cantonese_english = load_dataset("AlienKevin/mixed_cantonese_and_english_speech", split='train',  trust_remote_code=True, cache_dir=cache_dir)
cantonese_english = cantonese_english.remove_columns(["topic"])
cantonese_english = cantonese_english.cast_column("audio", Audio(sampling_rate=16000)).shuffle(seed=42).shuffle(seed=64)

train_size = int(len(cantonese_english) * 0.8)

print(train_size)
print(len(cantonese_english) - train_size)

cantonese_english_train = cantonese_english.select(range(0, train_size))
cantonese_english_test = cantonese_english.select(range(train_size, len(cantonese_english)))

print('mixed train_records: ' + str(len(cantonese_english_train)))
print('mixed test_records: ' + str(len(cantonese_english_test)))

cantonese_daily = load_dataset("ziyou-li/cantonese_daily", split='train',  trust_remote_code=True, cache_dir=cache_dir)
cantonese_daily = cantonese_english.cast_column("audio", Audio(sampling_rate=16000)).shuffle(seed=42).shuffle(seed=64)

train_size_daily = int(len(cantonese_daily) * 0.8)

cantonese_daily_train = cantonese_english.select(range(0, train_size_daily))
cantonese_daily_test = cantonese_english.select(range(train_size_daily, len(cantonese_daily)))


common_voice["train"] = concatenate_datasets([common_voice["train"], cantonese_english_train, cantonese_daily_train]).shuffle(seed=42).shuffle(seed=64)
common_voice["test"] = concatenate_datasets([common_voice["test"], cantonese_english_test, cantonese_daily_test]).shuffle(seed=42).shuffle(seed=64)

print(common_voice["train"][0])
print('merged train_records: ' + str(len(common_voice["train"])))
print('merged test_records: ' + str(len(common_voice["test"])))

# %%
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# %%
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

model = WhisperForConditionalGeneration.from_pretrained(model_path, cache_dir=cache_dir)
model.generation_config.language = target_lang
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None


# %%
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# %%
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# %%
metric = evaluate.load("cer")

# %%
def compute_metrics(pred:  EvalPrediction):
    print('computing metrics')
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    pred_ids_sample = pred.predictions[:int(len(pred_ids)/10)]
    label_ids_sample = pred.label_ids[:int(len(label_ids)/10)]

    # label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids_sample[label_ids_sample == -100] = tokenizer.pad_token_id
    
    # start_full = time.time()
    # pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    # stop_full = time.time()

    # print("full decode: " + str(stop_full - start_full))


    start_sample = time.time()
    pred_str_sample = tokenizer.batch_decode(pred_ids_sample, skip_special_tokens=True)
    label_str_sample = tokenizer.batch_decode(label_ids_sample, skip_special_tokens=True)
    stop_sample = time.time()

    print("sample decode: " + str(stop_sample - start_sample))

    # start_full = time.time()
    # cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    # stop_full = time.time()
    # print("full compute: " + str(stop_full - start_full))

    start_sample = time.time()
    cer_sample = 100 * metric.compute(predictions=pred_str_sample, references=label_str_sample)
    stop_sample = time.time()

    print("sample compute: " + str(stop_sample - start_sample))

    # print("sample vs full: " + str(cer_sample) + ": " + str(cer))

    print('computed')
    # return {"cer": cer}
    return {"cer": cer_sample}

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=50000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=64,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    report_to=["tensorboard"],
    # load_best_model_at_end=True,
    # metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# %%
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# %%
processor.save_pretrained(training_args.output_dir)

# %%
trainer.train()
# trainer.train(resume_from_checkpoint=True)


#%%
pipe = pipeline(task="automatic-speech-recognition", model=model, tokenizer=tokenizer)
try:
    pipe.save_pretrained(training_args.output_dir + '/final')
except:
    pipe.save_pretrained(training_args.output_dir + '/result')
