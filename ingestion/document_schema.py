from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    """
    Metadata schema representing ingested documents.
    Strictly typed and validated using Pydantic.
    """
    document_id: str = Field(
        ...,
        description="Unique UUID generated for the document record"
    )
    filename: str = Field(
        ...,
        description="Name of the file"
    )
    upload_date: datetime = Field(
        ...,
        description="Timestamp of document upload"
    )
    file_type: str = Field(
        ...,
        description="Type of the file (extension e.g., pdf, docx, pptx, txt, md)"
    )
    file_size: int = Field(
        ...,
        description="Size of the file in bytes"
    )
    sha256_hash: str = Field(
        ...,
        description="SHA256 checksum hash of the file content"
    )
    source: str = Field(
        ...,
        description="Storage or path location of the file"
    )
    chunk_count: int = Field(
        default=0,
        description="Number of chunks generated from the document"
    )
    extra_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata values"
    )
