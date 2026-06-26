# Test Report: Quality Assurance & Regression Verification

This report details the execution of both unit testing and end-to-end automated integration testing for the Company Knowledge Base Q&A RAG System.

---

## 1. Test Strategy Overview
The RAG system was subjected to two distinct testing methodologies:
1. **Unit Testing (`pytest`):** Runs regression tests on individual modules, including configuration validators, ingestion pipelines, metadata registry, logging buffers, retrievers, query engines, and evaluation templates.
2. **End-to-End Integration Testing (`verify_ingestion.py`):** Simulates actual user behaviors by writing a physical PDF document, parsing it with `pypdf`, checking size/extension constraints, chunking with the sentence splitter, calling the Gemini Embeddings API, storing vectors in a persistent ChromaDB instance, querying via hybrid search, checking citation references, and testing conversation memory.

---

## 2. Regression Test Suite (`pytest`)
A total of 21 tests were executed inside the virtual environment:
```bash
$env:PYTHONPATH="."
.venv\Scripts\pytest
```

### Execution Summary
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\jaypo\OneDrive\Desktop\company-kb-rag
plugins: anyio-4.14.0, langsmith-0.9.0, mock-3.15.1
collected 21 items

tests\test_embeddings.py ....                                            [ 19%]
tests\test_evaluation.py .                                               [ 23%]
tests\test_ingestion.py .....                                            [ 47%]
tests\test_logging.py .                                                  [ 52%]
tests\test_query_engine.py ...                                           [ 66%]
tests\test_retrieval.py ..                                               [ 76%]
tests\test_settings.py .....                                             [100%]

============================= 21 passed in 8.00s ==============================
```

### Test Case Coverage Details
* **`test_embeddings.py`:** Verifies embedding initialization, batch processing, and query embedding correctness with mock endpoints.
* **`test_evaluation.py`:** Asserts the structure of RAGAS evaluation metrics and prompt formatting.
* **`test_ingestion.py`:** Tests the `MetadataRegistry` hash computation, duplicate protection, schema mapping, and `SentenceSplitter` constraints.
* **`test_logging.py`:** Ensures the Loguru configuration successfully formats logs and intercepts unhandled exceptions.
* **`test_query_engine.py`:** Confirms the conversation memory tracks message exchanges and query engine correctly constructs prompts.
* **`test_retrieval.py`:** Verifies hybrid retriever fusion scoring logic and score thresholds.
* **`test_settings.py`:** Validates environment variable parser types, port limits, and Pydantic validators.

---

## 3. End-to-End Automated Integration Verification

The integration script was run to simulate user uploading and Q&A interaction:
```bash
.venv\Scripts\python C:\Users\jaypo\.gemini\antigravity\scratch\verify_ingestion.py
```

### Log Output & Verification Verification
1. **Upload Received & Written to Disk:**
   `Writing test document: sample_test_doc.pdf` (saved to new default directory `./data/uploads/`).
2. **Loader / Parser Execution:**
   `Parsing content for 'sample_test_doc.pdf' using '.pdf' reader...` (pypdf reader initialization verified).
3. **Document Ingestion Validation:**
   `Ingestion successful for 'sample_test_doc.pdf'. Generated 1 raw pages/documents.` (returns non-empty Document objects).
4. **Sentence Chunker:**
   `Generated 1 chunk nodes.` (node count verified).
5. **Gemini Embedding Generation:**
   `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents` (embeddings generated for every node).
6. **ChromaDB Insertion:**
   `Writing 1 chunk nodes to ChromaDB...` (vectors stored in collection `company_knowledge_base`).
7. **Registry Metadata Saved:**
   `Registered new document: sample_test_doc.pdf with 1 chunks` (registry entry updated on disk).
8. **Interactive Q&A Retrieval:**
   `Querying LLM: 'Where are the corporate headquarters of InvisiaX located?'`
9. **Citation Verification:**
   `SUCCESS: Source citation verified successfully: sample_test_doc.pdf` (source documents correctly cited in RAG response).

---

## 4. QA Task Checklist Verification

| QA Task | Requirement | Status | Verification Details |
| :--- | :--- | :---: | :--- |
| **5** | Save Streamlit uploads to disk | **Passed** | Files written to disk under `./data/uploads` before parsing. |
| **6** | Supported Readers: PDF, DOCX, TXT, PPTX, MD, HTML | **Passed** | `HTMLTagReader` imported and registered for `.html`/`.htm`. |
| **7** | Loader returns >= 1 Document object | **Passed** | Loader throws exception if document collection is empty. |
| **8** | Chunker returns non-empty nodes | **Passed** | Chunk size validates positive and yields active chunks. |
| **9** | Embedding Count = Node Count | **Passed** | Ingestion verification script asserts matches. |
| **10** | Chroma collection count increases | **Passed** | Validated via direct `collection.count()` database query. |
| **11** | VectorStoreIndex persists | **Passed** | Chroma client automatically flushes vectors to directory on disk. |
| **12** | Retriever uses latest index | **Passed** | Query engine reloads index dynamically. |
| **13** | Q&A works immediately | **Passed** | Checked in integration verification query step. |
| **14** | st.session_state updates | **Passed** | Catalog timestamps and state flags refresh dynamically. |
| **15** | UI Progress Indicator | **Passed** | Streamlit `st.status()` traces all 10 stages in real time. |
| **17** | Stack trace on failure | **Passed** | Generic error messages replaced with `st.exception()` tracebacks. |
| **18** | Diagnostics Dashboard | **Passed** | Completed diagnostics tab compiling DB, Gemini, and storage health. |
