from typing import List, Dict, Optional
from loguru import logger

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode, MetadataFilters
from config.exceptions import RetrievalError

class KBRetriever(BaseRetriever):
    """
    Advanced Retriever for the Company Knowledge Base.
    Inherits from LlamaIndex's BaseRetriever.
    Coordinates similarity searches, keyword queries, hybrid ranking, and score thresholds.
    """
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 4,
        score_threshold: float = 0.25,
        hybrid: bool = True,
        filters: Optional[MetadataFilters] = None
    ) -> None:
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.score_threshold = score_threshold
        self.hybrid = hybrid
        self.filters = filters
        # Initialize BaseRetriever class internals
        super().__init__()
        logger.debug(
            "KBRetriever initialized: top_k={}, threshold={}, hybrid_search={}, filters={}",
            similarity_top_k,
            score_threshold,
            hybrid,
            filters is not None
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Internal retrieval interface invoked by LlamaIndex query engines.
        """
        query_str = query_bundle.query_str
        try:
            logger.info("Executing retrieval pipeline for query: '{}'", query_str)

            # 1. Semantic/Vector Retrieval (QueryMode.DEFAULT)
            vector_retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k,
                vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
                filters=self.filters
            )
            vector_results = vector_retriever.retrieve(query_bundle)
            
            # Apply score threshold on vector results
            if self.score_threshold > 0.0:
                vector_results = [
                    item for item in vector_results
                    if (item.score if item.score is not None else 0.0) >= self.score_threshold
                ]

            if not self.hybrid:
                logger.info("Semantic-only retrieval completed. Found {} nodes.", len(vector_results))
                return vector_results

            # 2. Keyword/Text Retrieval (QueryMode.TEXT_SEARCH)
            keyword_retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k,
                vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH,
                filters=self.filters
            )
            keyword_results = keyword_retriever.retrieve(query_bundle)

            # 3. Fuse Rankings using Reciprocal Rank Fusion (RRF)
            fused_results = self._reciprocal_rank_fusion(
                vector_results,
                keyword_results,
                top_k=self.similarity_top_k
            )

            # Log details
            for idx, item in enumerate(fused_results):
                name = item.node.metadata.get("filename", "unknown")
                logger.debug(
                    "Fused Result {} - Score: {:.4f} - Source: {}",
                    idx + 1,
                    item.score or 0.0,
                    name
                )

            logger.info("Hybrid retrieval completed. Returned {} fused nodes.", len(fused_results))
            return fused_results

        except Exception as e:
            logger.error("Failed to retrieve nodes for query '{}': {}", query_str, str(e))
            raise RetrievalError(
                message=f"Node retrieval failed: {str(e)}"
            ) from e

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[NodeWithScore],
        keyword_results: List[NodeWithScore],
        top_k: int,
        rrf_constant: int = 60
    ) -> List[NodeWithScore]:
        """
        Merges vector and keyword results using Reciprocal Rank Fusion (RRF).
        Calculates RRF score: score = sum(1 / (rank + rrf_constant))
        """
        fused_scores: Dict[str, float] = {}
        node_map: Dict[str, NodeWithScore] = {}

        def accumulate_scores(results: List[NodeWithScore]) -> None:
            for rank, item in enumerate(results):
                node_id = item.node.node_id
                node_map[node_id] = item
                # RRF Formula
                fused_scores[node_id] = fused_scores.get(node_id, 0.0) + (1.0 / (rank + rrf_constant))

        accumulate_scores(vector_results)
        accumulate_scores(keyword_results)

        # Sort candidates descending by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda nid: fused_scores[nid], reverse=True)

        fused_results = []
        for nid in sorted_ids[:top_k]:
            item = node_map[nid]
            item.score = fused_scores[nid]
            fused_results.append(item)

        return fused_results
