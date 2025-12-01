# Orca: Web Edition (Gemini Powered) ☁️

**Google 5-Day AI Agents Intensive – Capstone Project**
**Track:** Concierge Agents (Cloud Deployment Version)

## 🌊 Overview

This is the **Cloud/Web Version** of Orca. Unlike the standard version which uses local LLMs (Ollama), this version is powered entirely by **Google Gemini 1.5 Pro & Flash**.

**Why this version?**
- 🚀 **Deploy Anywhere**: Runs on Cloud Run, Hugging Face, Render, or any CPU-only server.
- ⚡ **No GPU Needed**: All heavy lifting is done by the Gemini API.
- 🧠 **Simulated Experts**: The "Panel of Experts" (Mistral, Llama, Phi) are simulated using Gemini with specific system instructions/personas.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- A Google Cloud API Key (Get one at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone & Install
```bash
git clone https://github.com/Sansyuh06/stocks.git
cd "Orca but web"
pip install -r requirements.txt
```

### 2. Set API Key
You must set your Google API Key. You can do this in `app.py` or via environment variable.

**Option A: Environment Variable (Recommended)**
```bash
export GOOGLE_API_KEY="your_key_here"
# or on Windows:
set GOOGLE_API_KEY=your_key_here
```

**Option B: Hardcode in `app.py`**
Open `app.py` and find line ~1085:
```python
GOOGLE_API_KEY = 'your_key_here'
```

### 3. Run
```bash
python app.py
```
Access the dashboard at `http://localhost:5000`

---

## ☁️ Deployment

### Deploy to Google Cloud Run (Bonus Points! 🏆)

1. **Install Google Cloud SDK**
2. **Deploy**:
   ```bash
   gcloud run deploy orca-web --source . --set-env-vars GOOGLE_API_KEY=your_key
   ```

### Deploy to Hugging Face Spaces

1. Create a new Space (Docker).
2. Upload these files.
3. Add `GOOGLE_API_KEY` in Settings -> Secrets.

---

**Developer**: Sansyuh06 | **License**: MIT
 
