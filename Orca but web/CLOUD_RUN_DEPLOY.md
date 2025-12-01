#  Deploying Orca to Google Cloud Run

Follow these exact steps to deploy your app and get the **5 Bonus Points**!

## Prerequisites
1.  **Google Cloud Account**: [console.cloud.google.com](https://console.cloud.google.com)
2.  **Google Cloud SDK**: Installed on your machine ([Guide](https://cloud.google.com/sdk/docs/install))
3.  **Gemini API Key**: From [aistudio.google.com](https://aistudio.google.com)

---

## Step 1: Login & Configure
Open your terminal in this folder (`Orca but web`) and run:

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID (create one in the console if you haven't)
gcloud config set project YOUR_PROJECT_ID
```

## Step 2: Deploy
Run this single command to build and deploy. Replace `YOUR_API_KEY` with your actual key.

```bash
gcloud run deploy orca-web \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=YOUR_API_KEY
```

## Step 3: Success!
The command will finish and show a URL like:
`https://orca-web-xyz123-uc.a.run.app`

Click it, and your app is live! 

---

## Troubleshooting
*   **"Billing not enabled"**: You need to enable billing on your Google Cloud project (Cloud Run has a generous free tier).
*   **"API not enabled"**: The command will ask to enable the "Cloud Run API" and "Artifact Registry API". Type `y` to accept.
