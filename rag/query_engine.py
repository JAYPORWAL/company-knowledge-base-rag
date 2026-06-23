from typing import List, Dict, Any, Generator
from loguru import logger

from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.response.schema import Response, StreamingResponse
from config.exceptions import RetrievalError
from rag.retriever import KBRetriever

class ConversationMemoryPlaceholder:
    """
    Session-based conversation memory manager.
    Tracks user-assistant dialogue history to guide contextual Q&A.
    """
    def __init__(self, session_id: str = "default_session") -> None:
        self.session_id = session_id
        self._history: List[Dict[str, str]] = []
        logger.debug("ConversationMemoryPlaceholder initialized for session: '{}'", session_id)

    def add_message(self, role: str, content: str) -> None:
        """Add user or assistant message to memory history."""
        self._history.append({"role": role, "content": content})
        logger.debug("Logged message in memory: role={}, size={} chars", role, len(content))

    def get_history(self) -> List[Dict[str, str]]:
        """Retrieve complete session chat history."""
        return self._history

    def get_compiled_history(self) -> str:
        """Compile the last few turns of conversation for prompt insertion."""
        if not self._history:
            return "No previous conversation history."
        
        lines = []
        # Inject the last 3 turns (6 messages) to maintain prompt size constraints
        for msg in self._history[-6:]:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear_history(self) -> None:
        """Clear memory cache."""
        self._history.clear()
        logger.info("Cleared conversation history for session: '{}'", self.session_id)


class StreamingResponseWrapper:
    """
    Container wrapping LlamaIndex StreamingResponse operations.
    Allows downstream applications to stream tokens chunk-by-chunk while preserving
    citations and final token usage counts.
    """
    def __init__(self, raw_response: StreamingResponse, memory_manager: ConversationMemoryPlaceholder, query_str: str) -> None:
        self.raw_response = raw_response
        self.memory_manager = memory_manager
        self.query_str = query_str
        self.answer_accumulator: List[str] = []
        self.citations = self._extract_citations(raw_response)
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _extract_citations(self, response: StreamingResponse) -> List[Dict[str, Any]]:
        citations = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for idx, source_node in enumerate(response.source_nodes):
                metadata = source_node.node.metadata
                score = source_node.score if source_node.score is not None else 0.0
                citations.append({
                    "citation_number": idx + 1,
                    "filename": metadata.get("filename", "unknown"),
                    "file_type": metadata.get("file_type", "unknown"),
                    "sha256_hash": metadata.get("sha256_hash", "unknown"),
                    "source_path": metadata.get("source", "unknown"),
                    "score": score,
                    "snippet": source_node.node.get_content().strip()
                })
        return citations

    def response_generator(self) -> Generator[str, None, None]:
        """
        Generator yielding text chunks as they arrive from the LLM.
        Once the stream finishes, updates the conversation memory and token usage.
        """
        try:
            # Yield tokens from LlamaIndex StreamingResponse generator
            for chunk in self.raw_response.response_gen:
                self.answer_accumulator.append(chunk)
                yield chunk
            
            # Post-stream operations: accumulate response text and update memory
            full_answer = "".join(self.answer_accumulator)
            self.memory_manager.add_message("user", self.query_str)
            self.memory_manager.add_message("assistant", full_answer)

            # Retrieve final token metrics
            self.token_usage = self._extract_token_usage(self.raw_response)
            logger.info("Streaming complete. Final token counts: {}", self.token_usage)
            
        except Exception as e:
            logger.error("Token streaming failed: {}", str(e))
            raise RetrievalError(
                message=f"Failed to stream response token: {str(e)}"
            ) from e

    def _extract_token_usage(self, response: StreamingResponse) -> Dict[str, int]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            if hasattr(response, "metadata") and response.metadata:
                for key, val in response.metadata.items():
                    if isinstance(val, dict):
                        p = val.get("prompt_tokens") or val.get("input_tokens") or val.get("prompt_token_count")
                        c = val.get("completion_tokens") or val.get("output_tokens") or val.get("candidates_token_count")
                        t = val.get("total_tokens") or val.get("total_token_count")
                        if p or c:
                            usage["prompt_tokens"] = int(p or 0)
                            usage["completion_tokens"] = int(c or 0)
                            usage["total_tokens"] = int(t or (usage["prompt_tokens"] + usage["completion_tokens"]))
                            return usage
                    if "prompt_tokens" in key or "input_tokens" in key:
                        usage["prompt_tokens"] = int(val)
                    elif "completion_tokens" in key or "output_tokens" in key:
                        usage["completion_tokens"] = int(val)
                    elif "total_tokens" in key:
                        usage["total_tokens"] = int(val)
            
            if usage["total_tokens"] == 0:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        except Exception as e:
            logger.debug("Failed to extract token usage from streaming response: {}", str(e))
        return usage


class RAGQueryEngine:
    """
    Enterprise-grade Query Engine.
    Orchestrates advanced retrieval, prompt assembly with memory,
    streaming/synchronous execution, citation compilation, and token tracking.
    """
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 4,
        score_threshold: float = 0.25,
        hybrid: bool = True
    ) -> None:
        self.index = index
        self.retriever = KBRetriever(
            index,
            similarity_top_k=similarity_top_k,
            score_threshold=score_threshold,
            hybrid=hybrid
        )
        self.memory = ConversationMemoryPlaceholder()

        # Context-grounded QA prompt with conversation history integration
        self.qa_template = PromptTemplate(
            "You are a Senior Knowledge Management Assistant at our company.\n"
            "Your goal is to answer the user question factually using ONLY the provided company knowledge base context.\n"
            "Keep the response professional, concise, and accurate.\n"
            "If the answer cannot be found in the context documents, reply clearly with: "
            "'I am sorry, but I cannot find that information in the provided company documents.'\n"
            "Do not mention any information not present in the contexts below.\n\n"
            "Recent Conversation History:\n"
            "---------------------\n"
            "{history_str}\n"
            "---------------------\n\n"
            "Context Information:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"
            "Question: {query_str}\n"
            "Answer: "
        )

    def _get_query_engine(self, history_str: str, streaming: bool = False) -> RetrieverQueryEngine:
        """Constructs a LlamaIndex RetrieverQueryEngine pre-filled with history context."""
        partial_qa_template = self.qa_template.partial_format(history_str=history_str)
        return RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            text_qa_template=partial_qa_template,
            streaming=streaming
        )

    def query(self, query_str: str) -> Dict[str, Any]:
        """
        Synchronous Q&A query execution with full token tracking, citations, and memory logging.
        """
        try:
            logger.info("Executing synchronous enterprise query: '{}'", query_str)
            
            # Fetch compiled conversation history context
            history_str = self.memory.get_compiled_history()
            
            # Initialize query engine
            engine = self._get_query_engine(history_str, streaming=False)
            
            # Execute query
            response = engine.query(query_str)
            if not isinstance(response, Response):
                raise RetrievalError("Retrieved an unsupported response type from query engine.")

            answer_text = response.response or ""

            # Log to memory
            self.memory.add_message("user", query_str)
            self.memory.add_message("assistant", answer_text)

            # Compile citations
            citations = self._extract_citations(response)

            # Track token usage
            token_usage = self._extract_token_usage(response)
            logger.info("Query answered. Tokens used: {}", token_usage)

            return {
                "answer": answer_text,
                "citations": citations,
                "token_usage": token_usage,
                "history_count": len(self.memory.get_history())
            }

        except Exception as e:
            logger.error("Failed executing Q&A query: {}", str(e))
            raise RetrievalError(
                message=f"Q&A query execution failure: {str(e)}"
            ) from e

    def query_stream(self, query_str: str) -> StreamingResponseWrapper:
        """
        Initiates a streaming response, yielding a StreamingResponseWrapper container.
        """
        try:
            logger.info("Executing streaming enterprise query: '{}'", query_str)
            
            # Fetch compiled conversation history context
            history_str = self.memory.get_compiled_history()
            
            # Initialize query engine in streaming mode
            engine = self._get_query_engine(history_str, streaming=True)
            
            # Execute query
            response = engine.query(query_str)
            if not isinstance(response, StreamingResponse):
                raise RetrievalError("Retrieved an unsupported response type for streaming query.")

            # Wrap streaming response
            return StreamingResponseWrapper(response, self.memory, query_str)

        except Exception as e:
            logger.error("Failed initiating streaming query: {}", str(e))
            raise RetrievalError(
                message=f"Streaming query initiation failure: {str(e)}"
            ) from e

    def _extract_citations(self, response: Any) -> List[Dict[str, Any]]:
        citations = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for idx, source_node in enumerate(response.source_nodes):
                metadata = source_node.node.metadata
                score = source_node.score if source_node.score is not None else 0.0
                citations.append({
                    "citation_number": idx + 1,
                    "filename": metadata.get("filename", "unknown"),
                    "file_type": metadata.get("file_type", "unknown"),
                    "sha256_hash": metadata.get("sha256_hash", "unknown"),
                    "source_path": metadata.get("source", "unknown"),
                    "score": score,
                    "snippet": source_node.node.get_content().strip()
                })
        return citations

    def _extract_token_usage(self, response: Any) -> Dict[str, int]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            if hasattr(response, "metadata") and response.metadata:
                for key, val in response.metadata.items():
                    if isinstance(val, dict):
                        p = val.get("prompt_tokens") or val.get("input_tokens") or val.get("prompt_token_count")
                        c = val.get("completion_tokens") or val.get("output_tokens") or val.get("candidates_token_count")
                        t = val.get("total_tokens") or val.get("total_token_count")
                        if p or c:
                            usage["prompt_tokens"] = int(p or 0)
                            usage["completion_tokens"] = int(c or 0)
                            usage["total_tokens"] = int(t or (usage["prompt_tokens"] + usage["completion_tokens"]))
                            return usage
                    if "prompt_tokens" in key or "input_tokens" in key:
                        usage["prompt_tokens"] = int(val)
                    elif "completion_tokens" in key or "output_tokens" in key:
                        usage["completion_tokens"] = int(val)
                    elif "total_tokens" in key:
                        usage["total_tokens"] = int(val)
            
            if usage["total_tokens"] == 0:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        except Exception as e:
            logger.debug("Failed to extract token usage: {}", str(e))
        return usage
