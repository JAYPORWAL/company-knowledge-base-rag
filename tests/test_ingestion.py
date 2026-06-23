import pytest
from pathlib import Path
from llama_index.core import Document

from config.settings import Settings
from config.exceptions import IngestionError
from ingestion.metadata_registry import MetadataRegistry
from ingestion.parser import DocumentParserRegistry, TextFileReader
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker

def test_metadata_registry_operations(tmp_path: Path) -> None:
    """
    Verifies MetadataRegistry hashing, duplicate detection, and file persistence.
    """
    settings = Settings(
        GEMINI_API_KEY="test_gemini_key_ok",
        DATA_PROCESSED_DIR=str(tmp_path)
    )
    registry = MetadataRegistry(settings)
    
    # 1. Create temporary file to hash
    test_file = tmp_path / "policy.txt"
    test_file.write_text("Corporate Remote Work policy content", encoding="utf-8")
    
    # 2. Check hashing
    file_hash = registry.compute_sha256(test_file)
    assert len(file_hash) == 64
    
    # 3. Check duplicate detection
    assert not registry.is_duplicate(file_hash)
    
    # 4. Register document
    registry.register_document(
        document_id="doc-uuid-1",
        filename="policy.txt",
        file_type="txt",
        sha256_hash=file_hash,
        source=str(test_file),
        chunk_count=3
    )
    assert registry.is_duplicate(file_hash)
    
    # 5. Reload database from disk and verify checks remain active
    registry_reloaded = MetadataRegistry(settings)
    assert registry_reloaded.is_duplicate(file_hash)
    doc_record = registry_reloaded.get_document_by_hash(file_hash)
    assert doc_record is not None
    assert doc_record["chunk_count"] == 3


def test_text_file_reader(tmp_path: Path) -> None:
    """
    Verifies custom plain text file reader parsing logic.
    """
    reader = TextFileReader()
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("Example file content text.", encoding="utf-8")
    
    docs = reader.load_data(sample_file, extra_info={"document_id": "test-id"})
    assert len(docs) == 1
    assert docs[0].text == "Example file content text."
    assert docs[0].metadata["document_id"] == "test-id"


def test_document_parser_registry() -> None:
    """
    Verifies parser registration and resolution mappings.
    """
    registry = DocumentParserRegistry()
    
    # Check default reader resolution
    txt_reader = registry.get_reader_for_extension("txt")
    assert isinstance(txt_reader, TextFileReader)
    assert registry.get_reader_for_extension("non_existent") is None


def test_document_loader_constraints_and_validation(tmp_path: Path) -> None:
    """
    Verifies loader size checks, extension checks, and validation error propagation.
    """
    settings = Settings(
        GEMINI_API_KEY="test_gemini_key_ok",
        DATA_PROCESSED_DIR=str(tmp_path)
    )
    registry = MetadataRegistry(settings)
    
    # 1. Set loader with a strict 20-byte file size limit
    loader = DocumentLoader(settings, registry, max_file_size_bytes=20)

    # 2. Write invalid file format extension
    bad_ext_file = tmp_path / "script.py"
    bad_ext_file.write_text("print('hello')", encoding="utf-8")
    
    with pytest.raises(IngestionError) as excinfo:
        loader.load_file(bad_ext_file)
    assert "Unsupported file type" in str(excinfo.value)

    # 3. Write file exceeding size limit
    large_file = tmp_path / "large_file.txt"
    large_file.write_text("This content exceeds twenty bytes size limit.", encoding="utf-8")
    
    with pytest.raises(IngestionError) as excinfo:
        loader.load_file(large_file)
    assert "File exceeds maximum allowed size" in str(excinfo.value)


def test_document_chunker_sentence_splitting() -> None:
    """
    Verifies sentence splitter chunking, chunk ID mapping, and metadata preservation.
    """
    settings = Settings(
        GEMINI_API_KEY="test_gemini_key_ok",
        CHUNK_SIZE=50,
        CHUNK_OVERLAP=10
    )
    chunker = DocumentChunker(settings)
    
    # Construct LlamaIndex Document with metadata
    doc = Document(
        text="Sentence one. Sentence two. Sentence three. Sentence four.",
        metadata={
            "document_id": "doc-uuid-99",
            "filename": "sentences.txt"
        }
    )
    
    nodes = chunker.chunk_documents([doc], strategy="sentence")
    
    assert len(nodes) > 0
    # Assert chunk IDs are mapped sequentially
    assert nodes[0].metadata["chunk_id"] == "doc-uuid-99_chunk_00000"
    assert nodes[0].metadata["chunk_strategy"] == "sentence"
    # Assert metadata was preserved
    assert nodes[0].metadata["filename"] == "sentences.txt"
