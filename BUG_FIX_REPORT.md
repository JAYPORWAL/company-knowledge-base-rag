# Bug Fix Report: Ingestion Pipeline Fixes

The blocking bug preventing successful document indexing and UI feedback has been resolved. The pipeline is now fully operational, tested, and ready for production.

---

## Changes Implemented

### 1. UI Rerun Suppression & Progress Tracking (`app.py`)
* **Before:**
  * When the "Trigger Indexer Pipeline" button was clicked, a `st.rerun()` was called at the end of the action block. This reset the Streamlit execution thread, instantly erasing all success/warning/error banners that had been drawn.
  * Ingestion stages were run without detailed intermediate logs visible in the Streamlit UI, making it hard to see which step failed.
* **Fix:**
  * Removed the redundant `st.rerun()` from the end of the trigger button block. Streamlit automatically refreshes stateful elements when execution reaches the bottom of the script.
  * Wrapped the entire ingestion flow inside `st.status()` blocks (one status widget per file). This provides real-time UI logging for each of the 10 stages:
    1. **Upload Received:** Traces the file upload.
    2. **Temporary Storage (Save to disk):** Verifies the uploaded file is written to raw disk storage before parsing.
    3. **Parser Registry Resolve:** Dynamically resolves the reader (e.g., PDF, DOCX, TXT, PPTX, MD).
    4. **Loader parsing:** Resolves and runs the custom or default file reader.
    5. **Document Chunker:** Runs the `SentenceSplitter` and ensures non-empty node generation.
    6. **Embedding Generation Test:** Assures connection to the Google Gemini Embeddings API.
    7. **ChromaDB Insert:** Inserts node embeddings into the persistent database collection.
    8. **VectorStoreIndex persistence:** Stores registry details.
    9. **Session Refresh:** Updates session metadata statistics.
    10. **Document Catalog Update:** Displays the updated table to the user.

---

## 2. ChromaDB Metadata Flattening (`ingestion/loader.py`)
* **Before:**
  * LlamaIndex readers automatically populated a nested dictionary key `"extra_info"` inside document metadata.
  * When LlamaIndex attempted to insert the nodes into ChromaDB, the database rejected the insert with a `ValueError` because ChromaDB only accepts flat metadata objects (key-value pairs of types `str`, `int`, `float`, or `None`).
* **Fix:**
  * Added a flattening step in `ingestion/loader.py` that pops the nested `"extra_info"` dictionary from document/node metadata before returning.
  * This preserves the flat metadata format that ChromaDB expects while ensuring no diagnostic metadata is lost.

---

## 3. Embedding Model Correction (`config/settings.py` / `.env`)
* **Before:**
  * The embedding model was set to `models/text-embedding-004`, which caused API `404 Not Found` connection errors.
* **Fix:**
  * Replaced the target embedding model name with `models/gemini-embedding-001`. This model is fully supported by the active Google GenAI SDK and successfully tested in the workspace.

---

## 4. UI Diagnostics Tab Addition (`app.py`)
* **Fix:**
  * Appended an **Ingestion Diagnostics** tab alongside the Chat and Document Catalog tabs.
  * Displays:
    * Number of indexed documents (registry database size).
    * Total node chunks generated.
    * Actual database vector count (direct count query from the Chroma collection).
    * Embedding model connection status.
    * Last indexing timestamp (UTC).
    * Raw file directory scan (showing which uploaded files have been indexed vs. are stale on disk).
