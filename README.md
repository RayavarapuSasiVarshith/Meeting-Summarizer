# Meeting Summarizer (local)

Lightweight extractive meeting summarizer inspired by services like Fireflies.ai.

Files:
- `summarizer.py` — the summarization logic (TextRank-like extractive algorithm).
- `app.py` — minimal Flask web UI and API endpoint (`/api/summarize`).
- `run_tests.py` — small test runner to sanity check the summarizer.
- `requirements.txt` — Python dependencies (Flask).

Quick start (Windows PowerShell):

1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run tests:

```powershell
python run_tests.py
```

4. Start the web UI:

```powershell
python app.py
```

Then open http://127.0.0.1:5000/ and paste a meeting transcript, set the desired
number of sentences and click Summarize.

Notes and next steps:
- This is extractive (picks sentences from input). For abstractive summaries you'd
  need a model (transformers / OpenAI). This implementation is intentionally
  lightweight and offline.
- Improvements: better sentence tokenization, stopword removal, keyword highlighting,
  speaker diarization, audio-to-text pipeline (Whisper) and scheduled meeting ingestion.

Enabling OpenAI abstractive summaries (optional)
------------------------------------------------
To get high-quality abstractive paraphrases in your own words you can use OpenAI's API.

1. Install the library (already listed in `requirements.txt`):

```powershell
pip install -r requirements.txt
```

2. Set your OpenAI API key as an environment variable (PowerShell):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
```

Or set it permanently in your system environment variables.

3. When `OPENAI_API_KEY` is present the application will attempt to call OpenAI
   for abstractive summaries. If not present, it will fallback to a local
   transformers summarizer (if installed) or to a short extractive paraphrase.

Security note: Never commit your API keys to source control. Keep them in
environment variables or a secure secret store.

OpenAI (abstractive) support
--------------------------------
To enable higher-quality abstractive summaries using OpenAI's API:

1. Install Python dependencies (includes `openai` and `transformers`):

```powershell
pip install -r requirements.txt
```

2. Set your OpenAI API key in the environment (PowerShell):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
```

3. Restart the Flask app. The abstractive mode will automatically use OpenAI
   if the `OPENAI_API_KEY` environment variable is present and `openai` is
   installed. If OpenAI is not configured, the app falls back to a local
   transformers summarizer (if installed) or a short heuristic paraphrase.

Security note: keep your API key secret. Do not commit it to source control.
