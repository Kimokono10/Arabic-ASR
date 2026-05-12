"""
SCRIPT 1: Zero-Shot Baseline
Downloads pre-trained Whisper Small, evaluates on FLEURS Arabic (no training needed).
No GPU required. Takes ~5-10 minutes.

Run:
    python step1_zero_shot.py
"""

import os
import json
import re
import unicodedata
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

# ── Settings ───────────────────────────────────────────────────────────────────
MODEL_NAME     = "openai/whisper-small"
SAVE_PATH      = "./checkpoints/zero_shot_whisper_small"
RESULTS_PATH   = "./results/zero_shot_results.json"
NUM_EVAL_SAMPLES = 50

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# ── Arabic text normalization ──────────────────────────────────────────────────
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

# ── Step 1: Download and save model ───────────────────────────────────────────
print("=" * 55)
print("STEP 1: Downloading Whisper Small")
print("=" * 55)

model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="arabic", task="transcribe")
model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print(f"✅ Model saved to: {SAVE_PATH}")

# ── Step 2: Evaluate zero-shot on FLEURS Arabic ────────────────────────────────
print(f"\nSTEP 2: Zero-shot evaluation on FLEURS Arabic (n={NUM_EVAL_SAMPLES})")
print("=" * 55)

dataset = load_dataset("google/fleurs", "ar_eg",
                       split=f"test[:{NUM_EVAL_SAMPLES}]",
                       trust_remote_code=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

device = 0 if torch.cuda.is_available() else -1
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    generate_kwargs={"language": "arabic", "task": "transcribe"},
    device=device,
)

refs, hyps = [], []
print("Transcribing...")
for i, example in enumerate(dataset):
    audio = example["audio"]
    ref   = normalize(example["transcription"])
    result = asr_pipe({"array": audio["array"], "sampling_rate": audio["sampling_rate"]})
    hyp   = normalize(result["text"])
    refs.append(ref)
    hyps.append(hyp)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{NUM_EVAL_SAMPLES} done...")

total_wer = sum(compute_wer(r, h) for r, h in zip(refs, hyps)) / len(refs)
total_cer = sum(compute_cer(r, h) for r, h in zip(refs, hyps)) / len(refs)

results = {
    "condition": "zero_shot",
    "model": MODEL_NAME,
    "dataset": "google/fleurs Arabic (ar_eg)",
    "num_samples": NUM_EVAL_SAMPLES,
    "wer_percent": round(total_wer * 100, 2),
    "cer_percent": round(total_cer * 100, 2),
    "reference_wer_paper2": 23.76,
    "sample_predictions": [
        {"reference": refs[i], "hypothesis": hyps[i],
         "wer": round(compute_wer(refs[i], hyps[i]) * 100, 1)}
        for i in range(min(5, len(refs)))
    ]
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 55)
print("ZERO-SHOT RESULTS")
print("=" * 55)
print(f"  WER : {results['wer_percent']}%  (Paper 2 reference: 23.76%)")
print(f"  CER : {results['cer_percent']}%")
print(f"  Samples: {NUM_EVAL_SAMPLES}")
print(f"  Results saved: {RESULTS_PATH}")
print("=" * 55)
print("\nNext: python step2_fine_tune.py --quick")
