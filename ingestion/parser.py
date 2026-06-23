from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import PDFReader, DocxReader, PptxReader, MarkdownReader

class TextFileReader(BaseReader):
    """
    Custom fallback reader for plain text (.txt) files.
    """
    def load_data(self, file_path: Path, extra_info: Optional[Dict[str, Any]] = None) -> List[Document]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            # Construct standard LlamaIndex Document
            doc = Document(text=text, metadata=extra_info or {})
            return [doc]
        except Exception as e:
            logger.error("TextFileReader failed for file '{}': {}", file_path.name, str(e))
            raise


class DocumentParserRegistry:
    """
    Orchestrates the resolution and mapping of specific document extensions
    to LlamaIndex BaseReader components.
    """
    def __init__(self) -> None:
        self._parsers: Dict[str, BaseReader] = {}
        self._initialize_default_parsers()

    def _initialize_default_parsers(self) -> None:
        """
        Initializes core LlamaIndex document readers with fallback mechanisms.
        """
        # Register Plain Text Reader
        self._parsers["txt"] = TextFileReader()
        logger.debug("Registered custom TextFileReader for .txt")

        # Register PDF Reader
        try:
            self._parsers["pdf"] = PDFReader()
            logger.debug("Registered PDFReader for .pdf")
        except Exception as e:
            logger.warning("Failed to initialize PDFReader: {}", str(e))

        # Register Word Document Reader
        try:
            self._parsers["docx"] = DocxReader()
            logger.debug("Registered DocxReader for .docx")
        except Exception as e:
            logger.warning("Failed to initialize DocxReader: {}", str(e))

        # Register PowerPoint Reader
        try:
            self._parsers["pptx"] = PptxReader()
            logger.debug("Registered PptxReader for .pptx")
        except Exception as e:
            logger.warning("Failed to initialize PptxReader: {}", str(e))

        # Register Markdown Reader
        try:
            self._parsers["md"] = MarkdownReader()
            logger.debug("Registered MarkdownReader for .md")
        except Exception as e:
            logger.warning("Failed to initialize MarkdownReader: {}", str(e))

    def get_reader_for_extension(self, extension: str) -> Optional[BaseReader]:
        """
        Returns the mapped reader for the given extension (case-insensitive).
        """
        ext = extension.lower().lstrip(".")
        return self._parsers.get(ext)

    def register_parser(self, extension: str, reader: BaseReader) -> None:
        """
        Registers a new custom reader for a specific extension.
        Conforms to the Open-Closed Principle.
        """
        ext = extension.lower().lstrip(".")
        self._parsers[ext] = reader
        logger.info("Custom parser successfully registered for extension: '.{}'", ext)
