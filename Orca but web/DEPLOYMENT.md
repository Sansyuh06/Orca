# Deployment Guide for Orca

Since Orca relies on **local LLMs (Ollama)** and **GPU acceleration**, deploying it requires a different strategy than standard web apps. Here are 3 ways to deploy it, ranging from "Easy" to "Advanced".

---

## Option 1: The "Demo" Deployment (Ngrok) 
**Best for:** Recording your demo video, sharing with friends temporarily.
**Cost:** Free
**Difficulty:** Easy

This method keeps the app running on your powerful local machine but creates a secure tunnel so anyone on the internet can access it.

### Steps:
1.  **Download Ngrok**: [https://ngrok.com/download](https://ngrok.com/download)
2.  **Start Orca**:
    ```bash
    python app.py
    ```
3.  **Start Tunnel**:
    Open a new terminal and run:
    ```bash
    ngrok http 5000
    ```
4.  **Share**: Copy the `https://xxxx.ngrok-free.app` link. That's your public URL!

---

## Option 2: The "Cloud" Deployment (Google Cloud Run) 
**Best for:** Earning the "Agent Deployment" bonus points (5 pts).
**Cost:** Free Tier (mostly)
**Difficulty:** Advanced (Requires Code Changes)

**The Challenge**: Cloud Run is "serverless" (no GPU). It cannot run Ollama.
**The Fix**: You must modify `app.py` to use **Google Gemini API** instead of Ollama for the "Panel of Experts".

### Steps:
1.  **Modify `app.py`**:
    - Update `LLMOrchestrator` to check for an environment variable `DEPLOY_ENV`.
    - If `DEPLOY_ENV == 'cloud'`, call `genai.GenerativeModel('gemini-pro')` instead of `requests.post(OLLAMA_URL)`.
2.  **Containerize**:
    - Create a `Dockerfile` (see below).
3.  **Deploy**:
    ```bash
    gcloud run deploy orca --source . --set-env-vars GOOGLE_API_KEY=your_key,DEPLOY_ENV=cloud
    ```

### Sample Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

---

## Option 3: The "Power" Deployment (GPU VM) 
**Best for:** A permanent, fully functional clone of your local setup.
**Cost:** ~$0.50/hour
**Difficulty:** Medium

Rent a virtual machine with a GPU (e.g., NVIDIA T4).

### Providers:
- **Google Cloud Compute Engine** (N1 + T4 GPU)
- **Lambda Labs**
- **Paperspace**

### Steps:
1.  **Rent VM**: Select an Ubuntu instance with a GPU.
2.  **Install Drivers**: Install NVIDIA drivers and CUDA.
3.  **Install Ollama**: `curl -fsSL https://ollama.com/install.sh | sh`
4.  **Clone & Run**:
    ```bash
    git clone https://github.com/Sansyuh06/stocks.git
    pip install -r requirements.txt
    python app.py
    ```

---

## Option 4: The "AI Community" Way (Hugging Face Spaces) 
**Best for:** Public portfolio, easy sharing, "Agent Deployment" bonus.
**Cost:** Free (CPU) or Paid (GPU)
**Difficulty:** Medium

Deploy your app to Hugging Face, the home of the open-source AI community.

### The Catch:
Like Cloud Run, the **Free Tier is CPU-only**. You must use the **Gemini API fallback** (see Option 2) unless you pay for a GPU Space.

### Steps:
1.  **Create Space**: Go to [huggingface.co/spaces](https://huggingface.co/spaces) -> Create new Space -> Select "Docker".
2.  **Upload Code**: Upload your project files (including the `Dockerfile` from Option 2).
3.  **Set Secrets**: In Settings -> Repository secrets, add `GOOGLE_API_KEY`.
4.  **Run**: Hugging Face handles the building and hosting automatically.

---

## Option 5: The "Home Network" Way (LAN) 
**Best for:** Showing family/colleagues in the same office/house.
**Cost:** Free
**Difficulty:** Very Easy

Access the app from your phone or another laptop on the same WiFi.

### Steps:
1.  **Find your IP**:
    - Windows: Run `ipconfig` in terminal. Look for "IPv4 Address" (e.g., `192.168.1.15`).
2.  **Run with Host**:
    - Modify the last line of `app.py`:
      ```python
      app.run(host='0.0.0.0', port=5000, debug=True)
      ```
3.  **Access**: On your phone, go to `http://192.168.1.15:5000`.
