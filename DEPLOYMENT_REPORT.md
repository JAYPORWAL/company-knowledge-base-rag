# Deployment Report: Production Architecture & Guidelines

This document outlines the deployment configuration, environment specifications, and setup instructions to deploy the Company Knowledge Base Q&A RAG System in production environments.

---

## 1. Environment Specifications

* **Target OS:** Ubuntu 24.04 LTS (Fully compatible; can also run on Windows/Linux host environments).
* **Python Runtime:** Python 3.12.x or Python 3.13.x (Pydantic and LlamaIndex packages are pinned to stable releases).
* **Process Manager:** Systemd (for VM hosting) or Docker (container orchestration).
* **UI Server:** Headless Streamlit Server.

---

## 2. Configuration & Deployment Files

The application contains the following configurations tuned for production:

1. **`runtime.txt`:** Specifies `python-3.12` to ensure consistent runtime versioning on hosting providers like Streamlit Cloud.
2. **`.streamlit/config.toml`:**
   ```toml
   [server]
   headless = true
   
   [browser]
   gatherUsageStats = false
   ```
3. **`requirements.txt`:** Pinned versions of LlamaIndex core, Gemini embeddings, and document parsers (`pypdf`, `docx2txt`, `python-pptx`, `llama-index-readers-file`).

---

## 3. Directory Layout & Storage Persistence

The application maintains the following state directories:
* **`data/uploads/`:** Stores uploaded raw files (PDF, DOCX, TXT, PPTX, MD, HTML).
* **`data/processed/`:** Stores the metadata registry database file (`processed_documents.json`).
* **`data/chromadb/`:** Stores the ChromaDB persistent vector collection.
* **`logs/`:** Stores rolling application logs (`app.log`).

All of these directories are ignored in `.gitignore` to prevent committing customer knowledge base data into repository control, and are automatically created at startup or during container builds.

---

## 4. Docker Containerization

A production-grade `Dockerfile` is provided for containerized deployments.

### Highlights
* **Base Image:** `python:3.12-slim` to minimize image size.
* **Dependency Installation:** Pip dependencies are installed using a cached layer step prior to copying the rest of the application files.
* **Directory Provisioning:** Directories `data/uploads`, `data/processed`, `data/chromadb`, and `logs` are created during the build phase.
* **Healthcheck Configuration:** Includes a native healthcheck curling Streamlit's internal health status endpoint (`_stcore/health`):
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=10s --start-period=8s --retries=3 \
      CMD curl --fail http://localhost:8501/_stcore/health || exit 1
  ```

---

## 5. Streamlit Community Cloud Deployments

To deploy on Streamlit Community Cloud:
1. Push the code to a GitHub repository.
2. In the Streamlit Cloud dashboard, create a new app and link the repository.
3. In **Advanced Settings**, copy the contents of `.streamlit/secrets.toml.example` into the **Secrets** text box and input your active `GEMINI_API_KEY`.
4. Launch the application. The system will read files from the repository's requirements list, provision the Python 3.12 virtual environment, and bind to the secrets.

---

## 6. System Diagnostics and Health Checks

The application includes a background healthchecker (`health_checker.check_all()`) displayed in the sidebar:
* **System Operational Badge:** Shows operational status.
* **Diagnostics Details Expander:** Shows green/red indicators for:
  * Gemini API connectivity.
  * ChromaDB persistent connection.
  * Upload folder write permissions.
  * Processed metadata directory sanity.
  * Log path permissions.
