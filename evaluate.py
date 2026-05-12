"""
EVALUATION MODULE
Runs WER/CER evaluation with per-dialect performance breakdown.
Generates a detailed error analysis report.

Usage:
    python evaluate.py                        # evaluate zero-shot model
    python evaluate.py --model ./checkpoints/fine_tuned_whisper_small
    python evaluate.py --compare              # compare zero-shot vs fine-tuned
"""

import os, re, json, argparse, unicodedata
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="./checkpoints/zero_shot_whisper_small")
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--compare", action="store_true", help="Compare zero-shot vs fine-tuned")
args = parser.parse_args()

os.makedirs("./results", exist_ok=True)

# ── Text normalization ─────────────────────────────────────────────────────────
DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
ALEF_MAP   = {'\u0622':'\u0627','\u0623':'\u0627','\u0625':'\u0627','\u0671':'\u0627'}

def normalize(text):
    text = unicodedata.normalize('NFC', text)
    text = DIACRITICS.sub('', text)
    text = re.sub(r'\u0640', '', text)
    for v, c in ALEF_MAP.items(): text = text.replace(v, c)
    text = re.sub(r'[،؛؟!"\'()*+,\-./:;<=>?@\[\]^_`{|}~]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── Edit distance ──────────────────────────────────────────────────────────────
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
    return edit_distance(r, h) / max(len(r), 1) * 100

def compute_cer(ref, hyp):
    r, h = list(ref.replace(' ','')), list(hyp.replace(' ',''))
    return edit_distance(r, h) / max(len(r), 1) * 100

def get_errors(ref, hyp):
    """Count substitutions, deletions, insertions."""
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    if n == 0: return m, 0, 0
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    # Backtrace
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + (0 if r[i-1]==h[j-1] else 1):
            if r[i-1] != h[j-1]: S += 1
            i -= 1; j -= 1
        elif j > 0 and d[i][j] == d[i][j-1] + 1:
            I += 1; j -= 1
        else:
            D += 1; i -= 1
    return S, D, I

# ── Model inference ────────────────────────────────────────────────────────────
def load_model(model_path):
    print(f"  Loading: {model_path}")
    proc  = WhisperProcessor.from_pretrained(model_path, language="arabic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, proc, device

def transcribe(model, proc, device, audio_array, sr=16000):
    inputs = proc.feature_extractor(audio_array, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            inputs.input_features.to(device),
            language="arabic", task="transcribe",
            num_beams=5, no_speech_threshold=0.6,
            logprob_threshold=-1.0, compression_ratio_threshold=2.4,
            temperature=0.0,
        )
    return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]

# ── Evaluate one model on one dataset ─────────────────────────────────────────
def evaluate_dataset(model, proc, device, dataset_name, lang, split, label, n):
    print(f"\n  Dataset : {label}")
    print(f"  Samples : {n}")

    ds = load_dataset(dataset_name, lang, split=f"{split}[:{n}]", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    text_col = "transcription" if "transcription" in ds.column_names else "sentence"

    refs, hyps = [], []
    total_S = total_D = total_I = total_words = 0

    for i, ex in enumerate(ds):
        ref = normalize(ex[text_col])
        hyp = normalize(transcribe(model, proc, device, ex["audio"]["array"]))
        refs.append(ref); hyps.append(hyp)
        S, D, I = get_errors(ref, hyp)
        total_S += S; total_D += D; total_I += I
        total_words += max(len(ref.split()), 1)
        if (i+1) % 10 == 0: print(f"    {i+1}/{n} done...")

    wer_val = sum(compute_wer(r,h) for r,h in zip(refs,hyps)) / len(refs)
    cer_val = sum(compute_cer(r,h) for r,h in zip(refs,hyps)) / len(refs)

    result = {
        "dataset": label,
        "num_samples": len(refs),
        "wer": round(wer_val, 2),
        "cer": round(cer_val, 2),
        "substitutions": total_S,
        "deletions": total_D,
        "insertions": total_I,
        "total_ref_words": total_words,
        "sub_rate": round(total_S/total_words*100, 2),
        "del_rate": round(total_D/total_words*100, 2),
        "ins_rate": round(total_I/total_words*100, 2),
        "samples": [
            {"reference": refs[i], "hypothesis": hyps[i],
             "wer": round(compute_wer(refs[i], hyps[i]), 1)}
            for i in range(min(5, len(refs)))
        ]
    }

    print(f"  WER : {result['wer']}%")
    print(f"  CER : {result['cer']}%")
    print(f"  S={total_S}  D={total_D}  I={total_I}")
    return result

# ── Datasets to evaluate (per-dialect breakdown) ───────────────────────────────
EVAL_DATASETS = [
    # (dataset_name,          lang,    split,  label,              dialect)
    ("google/fleurs",        "ar_eg", "test", "FLEURS (MSA)",     "MSA"),
    ("google/fleurs",        "ar_eg", "validation", "FLEURS Val", "MSA"),
]

# ── Run evaluation ─────────────────────────────────────────────────────────────
def run_evaluation(model_path, n_samples, label="Model"):
    print(f"\n{'='*55}")
    print(f"EVALUATING: {label}")
    print(f"{'='*55}")
    model, proc, device = load_model(model_path)
    all_results = []
    for ds_name, lang, split, ds_label, dialect in EVAL_DATASETS:
        try:
            r = evaluate_dataset(model, proc, device, ds_name, lang, split,
                                 f"{ds_label} [{dialect}]", n_samples)
            r["dialect"] = dialect
            all_results.append(r)
        except Exception as e:
            print(f"  ⚠️ Could not evaluate {ds_label}: {e}")
    return all_results


# ── Print summary table ────────────────────────────────────────────────────────
def print_table(results, model_label):
    print(f"\n{'='*65}")
    print(f"PER-DIALECT RESULTS — {model_label}")
    print(f"{'='*65}")
    print(f"{'Dataset':<25} {'Dialect':<12} {'WER%':<8} {'CER%':<8} {'S':<6} {'D':<6} {'I':<6}")
    print("-"*65)
    for r in results:
        print(f"{r['dataset']:<25} {r['dialect']:<12} "
              f"{r['wer']:<8} {r['cer']:<8} "
              f"{r['substitutions']:<6} {r['deletions']:<6} {r['insertions']:<6}")

    # Overall average
    avg_wer = sum(r['wer'] for r in results) / len(results)
    avg_cer = sum(r['cer'] for r in results) / len(results)
    print("-"*65)
    print(f"{'AVERAGE':<25} {'':<12} {round(avg_wer,2):<8} {round(avg_cer,2):<8}")
    print(f"{'='*65}")
    print("\nError type legend: S=Substitution  D=Deletion  I=Insertion")
    print(f"Paper 2 reference WER on FLEURS: 23.76% (Whisper Small zero-shot)")


# ── Main ───────────────────────────────────────────────────────────────────────
if args.compare:
    # Compare zero-shot vs fine-tuned
    zs_path = "./checkpoints/zero_shot_whisper_small"
    ft_path = "./checkpoints/fine_tuned_whisper_small"

    zs_results = run_evaluation(zs_path, args.samples, "ZERO-SHOT")
    print_table(zs_results, "Zero-Shot Whisper Small")

    if os.path.exists(ft_path):
        ft_results = run_evaluation(ft_path, args.samples, "FINE-TUNED")
        print_table(ft_results, "Fine-Tuned Whisper Small")

        # Side by side comparison
        print(f"\n{'='*55}")
        print("COMPARISON: Zero-Shot vs Fine-Tuned")
        print(f"{'='*55}")
        print(f"{'Dataset':<25} {'ZS WER%':<12} {'FT WER%':<12} {'Δ WER'}")
        print("-"*55)
        for zs, ft in zip(zs_results, ft_results):
            delta = round(zs['wer'] - ft['wer'], 2)
            arrow = "✅" if delta > 0 else "⚠️"
            print(f"{zs['dataset']:<25} {zs['wer']:<12} {ft['wer']:<12} {arrow} -{delta}%")
        print(f"{'='*55}")

        # Save comparison
        comparison = {
            "zero_shot": zs_results,
            "fine_tuned": ft_results,
        }
        with open("./results/dialect_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print("\n✅ Saved: ./results/dialect_comparison.json")
    else:
        print(f"\n⚠️ Fine-tuned model not found at {ft_path}")
        print("   Run step2_fine_tune.py first.")

else:
    # Single model evaluation
    results = run_evaluation(args.model, args.samples,
                             os.path.basename(args.model))
    print_table(results, args.model)

    out = {"model": args.model, "results": results}
    model_label = os.path.basename(args.model.rstrip("/\\"))
    out_path = f"./results/eval_{model_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {out_path}")
