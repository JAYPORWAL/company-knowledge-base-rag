# Deployment Guide - Streamlit Community Cloud

This guide provides step-by-step instructions to deploy the **Company Knowledge Base Q&A RAG System** to **Streamlit Community Cloud** with secure secrets management, automated dependency installs, and system configurations.

---

## 📋 Prerequisites

Before deploying, ensure you have:
1. A **GitHub Account** with your project repository pushed.
2. A **Google Gemini API Key** (obtainable via [Google AI Studio](https://aistudio.google.com/)).
3. A **Streamlit Community Cloud Account** (sign up at [share.streamlit.io](https://share.streamlit.io/) using your GitHub account).

---

## 🛠️ Deployment Steps

### Step 1: Push Code to GitHub

Streamlit Cloud pulls directly from GitHub. Ensure your local code is committed and pushed:

```bash
# Verify your remote origin is configured
git remote -v

# Stage all files including the new deployment configs
git add .

# Commit your changes
git commit -m "deploy: configure streamlit community cloud manifests"

# Push to your main branch
git push origin main
```

---

### Step 2: Deploy on Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click the **"New App"** button in the top-right corner.
3. Configure the deployment fields:
   * **Repository:** Select `company-knowledge-base-rag` from the dropdown.
   * **Branch:** Select `main`.
   * **Main file path:** Type `app.py`.
   * **App URL:** (Optional) Customize the domain prefix if desired.

---

### Step 3: Configure Streamlit Secrets

Since the application requires the `GEMINI_API_KEY` to query the LLM and generate embeddings, you must configure Streamlit Secrets:

1. In the **New App** creation form, click the **"Advanced Settings"** link (or if the app is already deployed, go to **Settings** -> **Secrets** from your app dashboard).
2. Copy the configuration template from [.streamlit/secrets.toml.example](file:///.streamlit/secrets.toml.example) and paste it into the **Secrets** text box.
3. Replace the placeholder values with your actual credentials:
   ```toml
   # Paste this into your Streamlit Cloud Secrets panel:
   GEMINI_API_KEY = "AIzaSy..." # Your actual Gemini API key
   MODEL_PROVIDER = "gemini"
   EMBEDDING_PROVIDER = "gemini"
   LLM_MODEL = "gemini-2.5-flash"
   EMBEDDING_MODEL = "models/gemini-embedding-001"

   # Local ChromaDB persistent path (runs within the cloud instance container)
   CHROMA_DB_PATH = "./data/chromadb"
   CHROMA_COLLECTION_NAME = "company_knowledge_base"

   # Ingestion configurations
   DATA_RAW_DIR = "./data/raw"
   DATA_PROCESSED_DIR = "./data/processed"
   CHUNK_SIZE = 512
   CHUNK_OVERLAP = 50

   # Logger settings
   LOG_LEVEL = "INFO"
   LOG_FILE_PATH = "./logs/app.log"
   LOG_ROTATION = "10 MB"
   ```
4. Click **"Save"**.
5. Click **"Deploy!"**. Streamlit will spin up a container, fetch `runtime.txt`, install dependencies from `requirements.txt`, inject your secrets, and launch the portal.

---

## 🔍 Deployment Readiness & Security Audit

Before deployment, verify the following checks:

### 1. Hardcoded Credentials check
* **Audit Result:** `PASSED`. No API keys or tokens are stored in the codebase. All runtime credentials are loaded dynamically through `config/settings.py` which prioritizes `st.secrets` and `os.environ`.

### 2. Absolute Path dependencies
* **Audit Result:** `PASSED`. All storage paths (Chroma database, document raw upload folder, processed metadata JSON, log files) are configured to use relative directories (`./data/` and `./logs/`) that resolve cleanly in the container's working directory.

### 3. Python 3.12 Compatibility
* **Audit Result:** `PASSED`. The target version is locked via `runtime.txt` and verified compatible with all package ranges in `requirements.txt`. The local test suite passes 100% under Python 3.12.3.

---

## 🚨 Troubleshooting

### 1. `ConfigurationError: Application configuration validation failed: GEMINI_API_KEY`
* **Cause:** The `GEMINI_API_KEY` was not configured or is set to a default placeholder in Streamlit Secrets.
* **Resolution:** Ensure the Streamlit Secrets panel is populated with your active Google Gemini key, and double-check that there are no typo spaces in the value.

### 2. Streamlit Cloud Out of Memory (OOM)
* **Cause:** Ingesting large document collections (e.g., hundreds of long PDFs) simultaneously can exceed the memory limits of the free Streamlit Cloud container tier.
* **Resolution:** 
  - Upload files in smaller batches.
  - The application's SHA256 deduplication registry automatically prevents re-embedding existing files, reducing memory load during subsequent index refreshes.

### 3. File Permissions / Write Access Errors
* **Cause:** Attempting to write logs or vector data outside of the application workspace.
* **Resolution:** Streamlit Cloud allows read/write access to the current directory where the app is cloned. Keep `CHROMA_DB_PATH`, `DATA_RAW_DIR`, and `LOG_FILE_PATH` within the relative `./` structure.

### 4. `ModuleNotFoundError` during setup
* **Cause:** Streamlit Cloud failed to parse dependencies or timed out during pip install.
* **Resolution:** Access the Streamlit logs terminal on the bottom-right of the dashboard. If a dependency failed, check that it is listed correctly in `requirements.txt`.
