"""
SCRIPT 2: Fine-Tuning on FLEURS Arabic
Fine-tunes Whisper Small on Arabic data and saves the improved checkpoint.

Run:
    python step2_fine_tune.py --quick   (300 samples, 3 epochs — fast)
    python step2_fine_tune.py           (1000 samples, 5 epochs — recommended)
    python step2_fine_tune.py --full    (all data, 10 epochs — best results)
"""

import os
import re
import json
import argparse
import unicodedata
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ── Arguments ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true")
parser.add_argument("--full",  action="store_true")
args = parser.parse_args()

if args.quick:
    MAX_TRAIN_SAMPLES = 300
    MAX_EVAL_SAMPLES  = 100
    NUM_EPOCHS        = 3
    BATCH_SIZE        = 2
    GRAD_ACCUM        = 8
    MODE_LABEL        = "quick (300 samples, 3 epochs)"
elif args.full:
    MAX_TRAIN_SAMPLES = None
    MAX_EVAL_SAMPLES  = 500
    NUM_EPOCHS        = 10
    BATCH_SIZE        = 8
    GRAD_ACCUM        = 2
    MODE_LABEL        = "full dataset, 10 epochs"
else:
    MAX_TRAIN_SAMPLES = 1000
    MAX_EVAL_SAMPLES  = 200
    NUM_EPOCHS        = 5
    BATCH_SIZE        = 4
    GRAD_ACCUM        = 4
    MODE_LABEL        = "medium (1000 samples, 5 epochs)"

MODEL_NAME     = "openai/whisper-small"
SAVE_PATH      = "./checkpoints/fine_tuned_whisper_small"
RESULTS_PATH   = "./results/fine_tuned_results.json"
ZERO_SHOT_PATH = "./results/zero_shot_results.json"
LEARNING_RATE  = 1e-5
WARMUP_STEPS   = 100

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

print("=" * 55)
print("ARABIC ASR — FINE-TUNING")
print("=" * 55)
print(f"  Model  : {MODEL_NAME}")
print(f"  Mode   : {MODE_LABEL}")
print(f"  Epochs : {NUM_EPOCHS}")
print(f"  GPU    : {'YES ✅' if torch.cuda.is_available() else 'NO ⚠️'}")
print("=" * 55)

# ── Text normalization ─────────────────────────────────────────────────────────
DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
ALEF_MAP   = {'\u0622':'\u0627','\u0623':'\u0627','\u0625':'\u0627','\u0671':'\u0627'}

def normalize(text):
    text = unicodedata.normalize('NFC', text)
    text = DIACRITICS.sub('', text)
    text = re.sub(r'\u0640', '', text)
    for v, c in ALEF_MAP.items():
        text = text.replace(v, c)
    text = re.sub(r'[،؛؟!"\'()*+,\-./:;<=>?@\[\]^_`{|}~]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── WER / CER ──────────────────────────────────────────────────────────────────
def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[n][m]

def compute_wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(len(r), 1)

def compute_cer(ref, hyp):
    r = list(ref.replace(' ', ''))
    h = list(hyp.replace(' ', ''))
    return edit_distance(r, h) / max(len(r), 1)

# ── Load dataset ───────────────────────────────────────────────────────────────
print("\n[1/5] Loading FLEURS Arabic dataset...")
from datasets import load_dataset, Audio

dataset = load_dataset("google/fleurs", "ar_eg", trust_remote_code=True)

if MAX_TRAIN_SAMPLES:
    dataset["train"] = dataset["train"].select(range(min(MAX_TRAIN_SAMPLES, len(dataset["train"]))))
if MAX_EVAL_SAMPLES:
    dataset["validation"] = dataset["validation"].select(range(min(MAX_EVAL_SAMPLES, len(dataset["validation"]))))
    dataset["test"]       = dataset["test"].select(range(min(MAX_EVAL_SAMPLES, len(dataset["test"]))))

keep = ["audio", "transcription"]
for split in dataset:
    drop = [c for c in dataset[split].column_names if c not in keep]
    dataset[split] = dataset[split].remove_columns(drop)

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

print(f"  Train : {len(dataset['train'])} samples")
print(f"  Val   : {len(dataset['validation'])} samples")
print(f"  Test  : {len(dataset['test'])} samples")

# ── Load model ─────────────────────────────────────────────────────────────────
print("\n[2/5] Loading Whisper model...")
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="arabic", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids     = None
model.config.suppress_tokens        = []
model.generation_config.language    = "arabic"
model.generation_config.task        = "transcribe"
model.generation_config.forced_decoder_ids = None

# Freeze CNN layers
model.model.encoder.conv1.requires_grad_(False)
model.model.encoder.conv2.requires_grad_(False)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params    : {total:,}")
print(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")

# ── Preprocess ─────────────────────────────────────────────────────────────────
print("\n[3/5] Preprocessing...")

def preprocess(example):
    audio    = example["audio"]
    features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features[0]
    labels   = processor.tokenizer(normalize(example["transcription"])).input_ids
    return {"input_features": features, "labels": labels}

dataset = dataset.map(preprocess, remove_columns=["audio", "transcription"],
                      num_proc=1, desc="Preprocessing")
print("  ✅ Done")

# ── Data collator ──────────────────────────────────────────────────────────────
@dataclass
class DataCollator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_feats  = [{"input_features": f["input_features"]} for f in features]
        batch        = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        label_feats  = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels       = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollator(
    processor=processor,
    decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.bos_token),
)

# ── Metrics ────────────────────────────────────────────────────────────────────
import evaluate as hf_evaluate
wer_metric = hf_evaluate.load("wer")
cer_metric = hf_evaluate.load("cer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize(s) for s in processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    return {
        "wer": round(wer_metric.compute(predictions=pred_str, references=label_str) * 100, 2),
        "cer": round(cer_metric.compute(predictions=pred_str, references=label_str) * 100, 2),
    }

# ── Training ───────────────────────────────────────────────────────────────────
print("\n[4/5] Training...")
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

training_args = Seq2SeqTrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    logging_steps=20,
    predict_with_generate=True,
    generation_max_length=225,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

train_result = trainer.train()
print(f"\n✅ Training complete! Loss: {train_result.training_loss:.4f}")

trainer.save_model(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print(f"   Saved to: {SAVE_PATH}")

# ── Evaluate on test set ───────────────────────────────────────────────────────
print("\n[5/5] Evaluating on test set...")
from transformers import pipeline as hf_pipeline

asr_pipe = hf_pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    generate_kwargs={"language": "arabic", "task": "transcribe"},
    device=0 if torch.cuda.is_available() else -1,
)

test_raw = load_dataset("google/fleurs", "ar_eg",
                        split=f"test[:{MAX_EVAL_SAMPLES or 200}]",
                        trust_remote_code=True)
test_raw = test_raw.cast_column("audio", Audio(sampling_rate=16000))

refs, hyps = [], []
for i, example in enumerate(test_raw):
    ref    = normalize(example["transcription"])
    result = asr_pipe({"array": example["audio"]["array"],
                       "sampling_rate": example["audio"]["sampling_rate"]})
    hyp    = normalize(result["text"])
    refs.append(ref)
    hyps.append(hyp)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(test_raw)} done...")

ft_wer = sum(compute_wer(r, h) for r, h in zip(refs, hyps)) / len(refs)
ft_cer = sum(compute_cer(r, h) for r, h in zip(refs, hyps)) / len(refs)

# Load zero-shot for comparison
zs_wer, zs_cer = None, None
if os.path.exists(ZERO_SHOT_PATH):
    with open(ZERO_SHOT_PATH, encoding="utf-8") as f:
        zs = json.load(f)
        zs_wer = zs.get("wer_percent")
        zs_cer = zs.get("cer_percent")

improvement = round(zs_wer - ft_wer * 100, 2) if zs_wer else None

results = {
    "condition": "fine_tuned",
    "model": MODEL_NAME,
    "mode": MODE_LABEL,
    "dataset": "google/fleurs Arabic (ar_eg)",
    "num_train_samples": len(dataset["train"]),
    "num_epochs": NUM_EPOCHS,
    "num_test_samples": len(refs),
    "wer_percent": round(ft_wer * 100, 2),
    "cer_percent": round(ft_cer * 100, 2),
    "zero_shot_wer_for_comparison": zs_wer,
    "zero_shot_cer_for_comparison": zs_cer,
    "improvement_wer": improvement,
    "sample_predictions": [
        {"reference": refs[i], "hypothesis": hyps[i],
         "wer": round(compute_wer(refs[i], hyps[i]) * 100, 1)}
        for i in range(min(5, len(refs)))
    ]
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 55)
print("FINE-TUNING RESULTS")
print("=" * 55)
if zs_wer:
    print(f"  Zero-shot WER  : {zs_wer}%")
print(f"  Fine-tuned WER : {results['wer_percent']}%")
if improvement:
    print(f"  Improvement    : -{improvement}% WER ✅")
print(f"  Fine-tuned CER : {results['cer_percent']}%")
print(f"  Results saved  : {RESULTS_PATH}")
print("=" * 55)
print("\nNext: python app.py")
