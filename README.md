### 🗡️ Summaraii

Cut the Clutter, Keep the Core.

Summaraii is a Streamlit app that helps creators quickly extract topic-relevant insights from YouTube videos, websites, and PDFs. It filters each source by your topic first, then summarizes only what matters using an LLM backed by Groq.

Production demo: `https://summaraii.streamlit.app/`

### Features

- **Multi‑source input:** YouTube URLs, website URLs, and PDFs.
- **Topic filtering first:** Content is filtered by your topic before summarization.
- **LLM‑powered summaries:** Uses Groq Gemma2‑9b‑It via LangChain.
- **Clean UI:** Streamlit app with a two‑column layout and status messaging.

### Quickstart

Prerequisites:
- Python 3.9+
- A Groq API key

Clone and install:
```bash
git clone https://github.com/<your-org-or-user>/Summaraii.git
cd Summaraii
python -m venv .venv && .venv\Scripts\activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure environment:
- In the app sidebar you can paste your key each run, or create a `.env` file with:
```bash
GROQ_API_KEY=your_groq_api_key
```

Run the app:
```bash
streamlit run cs.py
```
Open `http://localhost:8501` in your browser.

### Usage
- Enter a topic in the sidebar.
- Optionally add YouTube and website URLs (one per line) and/or upload PDFs.
- Click “Summarize Content”. The app filters by topic and displays a combined summary.

### Configuration
- Model: `Gemma2-9b-It` via `langchain-groq`.
- Chunking: `RecursiveCharacterTextSplitter` with 5000/500.
- Filtering: simple regex match of your topic (case‑insensitive) before summarization.

### Development
Recommended commands:
```bash
# Lint (optional: add flake8/ruff to your env first)
python -m pip install ruff flake8
ruff check . || true
flake8 || true

# Run Streamlit
streamlit run cs.py
```

### Contributing
Contributions are welcome! Please read `CONTRIBUTING.md` and follow the code of conduct in `CODE_OF_CONDUCT.md`. For security issues, see `SECURITY.md`.

### License
MIT License. See `LICENSE`.
