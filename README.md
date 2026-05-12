# Arabic ASR System — Phase 2

**Team:** Karim Mohamed (8237) · Youssef Mohamed (8164) · Mohamed Maher (8070)

**Paper:** N-Shot Benchmarking of Whisper on Diverse Arabic ASR — INTERSPEECH 2023

---

## Setup & Run (in order)

```
pip install -r requirements.txt

python step1_zero_shot.py
python step2_fine_tune.py --quick
python app.py
```

Open **http://127.0.0.1:7860**

---

## Supported Audio Formats
WAV · MP3 · FLAC · OGG · M4A

---

## Files
| File | Purpose |
|------|---------|
| `step1_zero_shot.py` | Download Whisper, evaluate zero-shot WER |
| `step2_fine_tune.py` | Fine-tune on FLEURS Arabic, save checkpoint |
| `app.py` | Gradio demo (upload/record/evaluate/results) |
| `requirements.txt` | Python dependencies |

## Results Location
- `results/zero_shot_results.json`
- `results/fine_tuned_results.json`
- `checkpoints/zero_shot_whisper_small/`
- `checkpoints/fine_tuned_whisper_small/`
