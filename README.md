# SafarSetu

SafarSetu is a local English-to-multilingual travel translator for visitors to India. Travelers enter or speak English, choose a destination state or union territory for a supported local-language suggestion, and can change the target language at any time. The 40 regional and world-language targets include 24 Indian languages and regional varieties; English is the fixed source. Language options are verified against the installed NLLB tokenizer. A locally run NLLB-200 model handles open-ended translations; the reviewed Hindi phrase book takes precedence for its exact English-to-Hindi matches. The existing Keras GRU/attention training script remains available for the course model comparison.

## What is included

- Responsive Flask interface with a locally stored scenic Unsplash background, a searchable target-language picker, categorized quick phrases, and a voice assistant that captures English speech, translates it, and reads the result aloud.
- Voice recognition uses the browser's Web Speech API when available and requires microphone permission and internet access; recognition may be processed by the browser's online speech service. Spoken translations use the browser's speech synthesis voices.
- The header includes a custom SafarSetu map-pin/bridge logo. The emergency-support panel links to Google Maps searches for police stations and hospitals, focused on the selected destination area without requesting device location. Map listings are third-party results and SafarSetu does not contact emergency services.
- Local AI translation using `facebook/nllb-200-distilled-600M`. Model weights are downloaded from Hugging Face on the first translation request and cached locally. Typed phrases are sent to the local model, not to an external translation API. Voice recognition may use the browser's online speech service.
- A 52-pair Hindi phrase book at `data/travel_phrases.csv`; exact Hindi matches work without loading the AI model.
- The Keras training script `train.py` for the course exercise: attention vs. no-attention models and BLEU evaluation on the Hindi starter data.

This is a course-project prototype, not a certified translation or emergency service. The NLLB checkpoint is licensed **CC-BY-NC** and is not released for production deployment; review its [model card](https://huggingface.co/facebook/nllb-200-distilled-600M) before use, and do not use it commercially without resolving the license. Its creators caution against domain-specific medical/legal uses. Verify important translations with a fluent speaker and contact local emergency services directly.

## Run the web app

Use Python 3.10+ and install the dependencies in a virtual environment. PyTorch and the model weights take substantial disk space and the model needs several GB of available memory; the model is only downloaded when the first non-phrasebook translation is requested.

On Windows, after setup, double-click `run_safarsetu.bat`. It starts the Flask server and opens the app in your browser. Keep the server window open while using SafarSetu.

Do not open `templates/index.html` directly: it is a Flask/Jinja template, not a standalone HTML file. If opened directly, it explains how to launch the app. The app needs Flask to render it and serve translation requests and static assets.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, choose your destination area for a language suggestion (or select a target language directly), enter or speak an English phrase, and translate. Area suggestions are a starting point only; many areas are multilingual and not every local language is included. The first AI translation requires an internet connection to download the model. Once cached, model inference runs locally. Exact Hindi phrase-book matches do not require the model.

## Train and evaluate

For the CO2 seq2seq/attention experiment, review and expand `data/travel_phrases.csv`, then run:

```powershell
python train.py --epochs 100
```

The script saves `artifacts/attention_model.keras`, the source and target tokenizers, and `artifacts/evaluation.json`. This experimental custom model is Hindi-only and is not the multilingual model used for open-ended web translations. BLEU scores on a small holdout are only a development signal, not evidence of emergency-use accuracy.

## Test

```powershell
python -m unittest discover -s tests -v
```
