"""
Arabic ASR Demo Application
Run: python app.py --model ./checkpoints/zero_shot_whisper_small
Open: http://127.0.0.1:7860
"""

import os, time, json, re, unicodedata
import numpy as np
import gradio as gr
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEFAULT_MODEL   = "./checkpoints/zero_shot_whisper_small"
ZERO_SHOT_JSON  = "./results/zero_shot_results.json"
FINE_TUNED_JSON = "./results/fine_tuned_results.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

model_loaded   = None
processor_loaded = None
current_model  = None

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

def wer(r, h):
    a, b = r.split(), h.split()
    return edit_distance(a, b) / max(len(a), 1) * 100

def cer(r, h):
    a, b = list(r.replace(' ','')), list(h.replace(' ',''))
    return edit_distance(a, b) / max(len(a), 1) * 100

# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(model_path):
    global model_loaded, processor_loaded, current_model
    if not model_path.strip():
        return "⚠️ Enter a model path.", ""
    try:
        processor_loaded = WhisperProcessor.from_pretrained(
            model_path, language="arabic", task="transcribe"
        )
        model_loaded = WhisperForConditionalGeneration.from_pretrained(model_path)
        model_loaded.to(DEVICE)
        model_loaded.eval()
        current_model = model_path
        total = sum(p.numel() for p in model_loaded.parameters())
        return f"✅ Loaded: `{model_path}`", f"**Params:** {total:,} | **Device:** {DEVICE.upper()}"
    except Exception as e:
        return f"❌ {e}", ""

# ── Core transcription (direct inference, no pipeline) ────────────────────────
def _transcribe_chunk(audio_array):
    """Transcribe a single audio chunk (must be under 30 seconds)."""
    inputs = processor_loaded.feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(DEVICE)
    with torch.no_grad():
        predicted_ids = model_loaded.generate(
            input_features,
            language="arabic",
            task="transcribe",
            num_beams=5,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=0.0,
        )
    return processor_loaded.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]


def run_transcription(audio_array, sampling_rate):
    """
    Direct Whisper inference with anti-hallucination settings.
    Automatically chunks audio into 25-second windows for long files.
    """
    # Resample to 16kHz if needed
    if sampling_rate != 16000:
        import torchaudio.functional as F
        wav = torch.tensor(audio_array)
        wav = F.resample(wav, sampling_rate, 16000)
        audio_array = wav.numpy()

    CHUNK = 25 * 16000  # 25 seconds in samples
    duration = len(audio_array) / 16000

    # Short audio — transcribe directly
    if duration <= 25:
        return _transcribe_chunk(audio_array)

    # Long audio — chunk and join
    parts = []
    for start in range(0, len(audio_array), CHUNK):
        chunk = audio_array[start:start + CHUNK]
        if len(chunk) < 8000:  # skip chunks under 0.5s
            continue
        text = _transcribe_chunk(chunk).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


# ── Transcription handler ──────────────────────────────────────────────────────
def transcribe(audio_input, model_path, do_normalize):
    global model_loaded, processor_loaded, current_model
    if audio_input is None:
        return "", "", "⚠️ Please provide audio."
    if model_loaded is None or current_model != model_path:
        status, _ = load_model(model_path)
        if "❌" in status:
            return "", "", status

    t0 = time.time()
    sr, arr = audio_input
    arr = arr.astype(np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    mx = np.abs(arr).max()
    if mx > 0:
        arr /= mx

    try:
        text = run_transcription(arr, sr)
        if do_normalize:
            text = normalize(text)
    except Exception as e:
        return "", "", f"❌ {e}"

    elapsed  = time.time() - t0
    duration = len(arr) / sr
    info = (f"⏱️ **Duration:** {duration:.1f}s | "
            f"**Processing:** {elapsed:.2f}s | "
            f"**RTF:** {elapsed/duration:.2f}× | "
            f"**Device:** {DEVICE.upper()}")
    return text, info, ""

# ── Manual WER/CER ─────────────────────────────────────────────────────────────
def evaluate_manual(hyp, ref):
    if not hyp.strip(): return "⚠️ No transcription entered."
    if not ref.strip(): return "⚠️ No reference entered."
    h, r = normalize(hyp), normalize(ref)
    return (
        f"### 📊 Evaluation Results\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| **WER** | {wer(r,h):.2f}% |\n"
        f"| **CER** | {cer(r,h):.2f}% |\n"
        f"| Reference words | {len(r.split())} |\n"
        f"| Hypothesis words | {len(h.split())} |\n\n"
        f"*Lower is better. Human parity ≈ 8.9% WER on MGB-2.*"
    )

# ── Results tab ────────────────────────────────────────────────────────────────
def load_results():
    def rj(path):
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        return None

    zs = rj(ZERO_SHOT_JSON)
    ft = rj(FINE_TUNED_JSON)

    if not zs and not ft:
        return (
            "## 📊 Evaluation Results\n\n"
            "⚠️ **No results yet.**\n\n"
            "Run first:\n```\npython step1_zero_shot.py\n"
            "python step2_fine_tune.py --quick\n```\nThen click 🔄 Refresh."
        )

    md  = "## 📊 Evaluation Results\n\n"
    md += "### Zero-Shot vs Fine-Tuned\n\n"
    md += "| Condition | WER (%) | CER (%) | Samples |\n"
    md += "|-----------|---------|---------|--------|\n"
    if zs:
        md += f"| 🔵 Zero-Shot | **{zs.get('wer_percent','N/A')}** | {zs.get('cer_percent','N/A')} | {zs.get('num_samples','?')} |\n"
    if ft:
        md += f"| 🟢 Fine-Tuned | **{ft.get('wer_percent','N/A')}** | {ft.get('cer_percent','N/A')} | {ft.get('num_test_samples','?')} |\n"
    if ft and ft.get('improvement_wer'):
        md += f"\n✅ **Fine-tuning improved WER by {ft['improvement_wer']} points.**\n"

    md += "\n### Paper 2 Reference WERs (Whisper Small, Zero-Shot)\n\n"
    md += "| Dataset | Dialect | WER (%) |\n|---------|---------|--------|\n"
    md += "| FLEURS Arabic | MSA | 23.76 |\n"
    md += "| Common Voice 11 | Mixed | 19.41 |\n"
    md += "| MGB-2 | Broadcast | 34.74 |\n"
    md += "| MGB-3 | Egyptian | 49.68 |\n"
    md += "| MGB-5 | Moroccan Darija | 73.98 |\n"

    if zs and zs.get('sample_predictions'):
        md += "\n---\n### 🔵 Zero-Shot Sample Predictions\n\n"
        md += "| # | Reference | Hypothesis | WER |\n|---|-----------|------------|-----|\n"
        for i, s in enumerate(zs['sample_predictions'][:5], 1):
            md += f"| {i} | {s['reference']} | {s['hypothesis']} | {s['wer']}% |\n"

    if ft and ft.get('sample_predictions'):
        md += "\n---\n### 🟢 Fine-Tuned Sample Predictions\n\n"
        md += "| # | Reference | Hypothesis | WER |\n|---|-----------|------------|-----|\n"
        for i, s in enumerate(ft['sample_predictions'][:5], 1):
            md += f"| {i} | {s['reference']} | {s['hypothesis']} | {s['wer']}% |\n"

    md += "\n---\n### Error Analysis\n\n"
    md += "| Error Type | Description |\n|------------|-------------|\n"
    md += "| **Substitution** | Wrong word — most common in dialectal Arabic |\n"
    md += "| **Deletion** | Word missed entirely |\n"
    md += "| **Insertion** | Extra word added |\n"
    return md

# ── Architecture ───────────────────────────────────────────────────────────────
ARCH = """
## Whisper Architecture for Arabic ASR

```
Raw Audio (16kHz WAV)
      ↓
Log-Mel Spectrogram  [80 channels × T frames]
      ↓
┌──────────────────────────────────────┐
│  ENCODER  (12 Transformer layers)    │
│  Conv1D × 2  (stride 2, GELU)        │
│  Multi-head self-attention (6 heads) │
│  d_model = 768  |  ~120M params      │
└─────────────────┬────────────────────┘
                  │  cross-attention
┌─────────────────▼────────────────────┐
│  DECODER  (12 Transformer layers)    │
│  Forced: <|arabic|> <|transcribe|>   │
│  ~124M params                        │
└──────────────────────────────────────┘
      ↓
Arabic Text
```

| Model | Params | Enc | Dec | d_model |
|-------|--------|-----|-----|---------|
| Tiny  | 39M    | 4   | 4   | 384     |
| Base  | 74M    | 6   | 6   | 512     |
| **Small** | **244M** | **12** | **12** | **768** |
| Medium | 307M  | 24  | 24  | 1024    |
| Large | 1.55B  | 32  | 32  | 1280    |
"""

# ── Build UI ───────────────────────────────────────────────────────────────────
def build_demo():
    with gr.Blocks(title="Arabic ASR System") as demo:

        gr.Markdown("""
# 🎙️ Arabic Speech Recognition System
**Paper:** *N-Shot Benchmarking of Whisper on Diverse Arabic ASR* — INTERSPEECH 2023

**Team:** Karim Mohamed (8237) · Youssef Mohamed (8164) · Mohamed Maher (8070)

**Tip:** Use WAV files under 30 seconds for best results.
""")

        with gr.Accordion("⚙️ Model Settings", open=False):
            with gr.Row():
                model_input  = gr.Textbox(value=DEFAULT_MODEL, label="Model Path", scale=4)
                load_btn     = gr.Button("Load Model", variant="secondary", scale=1)
            model_status  = gr.Markdown("*Not loaded. Click Load Model to start.*")
            model_details = gr.Markdown()
            do_normalize  = gr.Checkbox(value=True, label="Normalize Arabic text (remove diacritics)")

        load_btn.click(load_model, [model_input], [model_status, model_details])

        with gr.Tabs():

            with gr.Tab("📁 Upload Audio"):
                with gr.Row():
                    with gr.Column():
                        up_audio = gr.Audio(label="Upload Audio (WAV recommended)",
                                            type="numpy", sources=["upload"])
                        up_btn   = gr.Button("🎯 Transcribe", variant="primary")
                    with gr.Column():
                        up_text  = gr.Textbox(label="Transcription (Arabic)", lines=6)
                        up_info  = gr.Markdown()
                        up_err   = gr.Markdown()
                up_btn.click(transcribe,
                             [up_audio, model_input, do_normalize],
                             [up_text, up_info, up_err])

            with gr.Tab("🎤 Record Speech"):
                with gr.Row():
                    with gr.Column():
                        mic_audio = gr.Audio(label="Record Arabic Speech",
                                             type="numpy", sources=["microphone"])
                        mic_btn   = gr.Button("🎯 Transcribe", variant="primary")
                    with gr.Column():
                        mic_text  = gr.Textbox(label="Transcription (Arabic)", lines=6)
                        mic_info  = gr.Markdown()
                        mic_err   = gr.Markdown()
                mic_btn.click(transcribe,
                              [mic_audio, model_input, do_normalize],
                              [mic_text, mic_info, mic_err])

            with gr.Tab("✏️ Evaluate (WER / CER)"):
                gr.Markdown("Paste a transcription and the correct reference to compute WER and CER.")
                with gr.Row():
                    eval_hyp = gr.Textbox(label="Model Output (Hypothesis)", lines=4)
                    eval_ref = gr.Textbox(label="Correct Text (Reference)",  lines=4)
                eval_btn    = gr.Button("Calculate WER / CER", variant="primary")
                eval_result = gr.Markdown()
                eval_btn.click(evaluate_manual, [eval_hyp, eval_ref], [eval_result])

            with gr.Tab("📊 Evaluation Results"):
                refresh_btn = gr.Button("🔄 Refresh Results", variant="secondary")
                results_md  = gr.Markdown(load_results())
                refresh_btn.click(fn=load_results, inputs=[], outputs=[results_md])

            with gr.Tab("🔬 Architecture"):
                gr.Markdown(ARCH)

        gr.Markdown("---\n**Refs:** Talafha et al. INTERSPEECH 2023 · Toyin et al. ArabicNLP 2023")
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    load_model(args.model)
    build_demo().launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
        inbrowser=True,
    )
