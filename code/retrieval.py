"""
Hybrid Embedding & Retrieval Pipeline
Implements FAISS + BM25 + Recency + Reranking + Clustering.
"""
import math
import logging
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.cluster import DBSCAN

from config import RetrievalConfig
from query_expansion import QueryExpander
from data_loader import DataLoader

logger = logging.getLogger(__name__)

class DummyModel:
    """Fallback for fast unit testing without loading real LLMs."""
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.random.rand(384).astype(np.float32)
        return np.random.rand(len(texts), 384).astype(np.float32)
        
    def predict(self, pairs, **kwargs):
        return np.random.rand(len(pairs)).astype(np.float32)
        
    def get_sentence_embedding_dimension(self) -> int:
        return 384
        
    def get_embedding_dimension(self) -> int:
        return 384

class EmbeddingStore:
    """Singleton-like cache for embeddings keyed by SHA-256."""
    _cache: Dict[str, np.ndarray] = {}
    
    @classmethod
    def get_key(cls, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
        
    @classmethod
    def get(cls, text: str) -> Optional[np.ndarray]:
        return cls._cache.get(cls.get_key(text))
        
    @classmethod
    def put(cls, text: str, embedding: np.ndarray):
        cls._cache[cls.get_key(text)] = embedding

class RetrievalEngine:
    def __init__(self, data_loader: DataLoader, use_mocks: bool = False):
        self.data_loader = data_loader
        self.config = RetrievalConfig()
        self.use_mocks = use_mocks
        
        logger.info("Initializing Retrieval Engine models...")
        if use_mocks:
            self.bi_encoder = DummyModel()
            if getattr(self.config, 'USE_CROSS_ENCODER', False):
                self.cross_encoder = DummyModel()
            self.embed_dim = 384
        else:
            # Using fast lightweight models for hackathon constraints
            self.bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            if getattr(self.config, 'USE_CROSS_ENCODER', False):
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            # SentenceTransformer dimensions
            try:
                self.embed_dim = self.bi_encoder.get_sentence_embedding_dimension()
            except Exception:
                self.embed_dim = self.bi_encoder.get_embedding_dimension()
            
        # Caches
        self.bm25_cache: Dict[str, BM25Okapi] = {}
        
        # Pre-build history index for O(1) isolation by user_id
        self.user_to_msgs: Dict[str, List[Dict[str, Any]]] = {}
        self.business_to_msgs: Dict[str, List[Dict[str, Any]]] = {}
        self.group_to_msgs: Dict[str, List[Dict[str, Any]]] = {}
        self._build_history_index()

    def _build_history_index(self):
        """Builds in-memory inverted indices for candidates to avoid O(N) global search."""
        history_dict = self.data_loader.data_dicts.get("message_history.csv", {})
        for msg_id, row in history_dict.items():
            row_w_id = dict(row)
            row_w_id["message_id"] = msg_id
            
            uid = str(row_w_id.get("user_id", ""))
            bid = str(row_w_id.get("business_id", ""))
            gid = str(row_w_id.get("group_id", ""))
            
            if uid and uid.lower() != "nan":
                self.user_to_msgs.setdefault(uid, []).append(row_w_id)
            if bid and bid.lower() != "nan":
                self.business_to_msgs.setdefault(bid, []).append(row_w_id)
            if gid and gid.lower() != "nan":
                self.group_to_msgs.setdefault(gid, []).append(row_w_id)

    def _get_candidates_unfiltered(self, msg_row: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Stage 1: Deterministic Filtering & Cold-Start Fallbacks. Returns cache key and candidates."""
        uid = str(msg_row.get("user_id", ""))
        ctype = str(msg_row.get("conversation_type", ""))
        
        candidates = self.user_to_msgs.get(uid, [])
        cache_key = f"user_{uid}"
        
        # Cold start fallback logic
        if not candidates:
            if ctype == "business":
                bid = str(msg_row.get("business_id", ""))
                candidates = self.business_to_msgs.get(bid, [])
                cache_key = f"business_{bid}"
            elif ctype == "group":
                gid = str(msg_row.get("group_id", ""))
                candidates = self.group_to_msgs.get(gid, [])
                cache_key = f"group_{gid}"
                
        return cache_key, candidates

    def _get_candidates(self, msg_row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Returns the filtered candidates list (used for unit tests / backward compatibility)."""
        _, all_cands = self._get_candidates_unfiltered(msg_row)
        current_msg_id = str(msg_row.get("message_id", ""))
        ctype = msg_row.get("conversation_type", "personal")
        filtered = []
        for c in all_cands:
            if str(c.get("message_id")) == current_msg_id:
                continue
            if ctype == "business" and str(c.get("business_id", "")) != str(msg_row.get("business_id", "")):
                continue
            filtered.append(c)
        return filtered

    def _get_text_payload(self, row: Dict[str, Any]) -> str:
        text = str(row.get("message_text", ""))
        ocr = str(row.get("ocr_text", ""))
        asr = str(row.get("asr_transcript", ""))
        parts = [p for p in (text, ocr, asr) if p and p.lower() != "nan"]
        return " ".join(parts)

    def _get_embeddings_batched(self, texts: List[str]) -> np.ndarray:
        results: List[Any] = []
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        
        for i, text in enumerate(texts):
            emb = EmbeddingStore.get(text)
            if emb is not None:
                results.append(emb)
            else:
                results.append(None)
                missing_indices.append(i)
                missing_texts.append(text)
                
        if missing_texts:
            new_embs = self.bi_encoder.encode(missing_texts)
            for idx, text, emb in zip(missing_indices, missing_texts, new_embs):
                results[idx] = emb
                EmbeddingStore.put(text, emb)
                
        return np.array(results)

    def _compute_recency_and_decay(self, hist_date_str: str, query_date_str: str) -> Tuple[float, float]:
        """Calculates independent recency score [0-1] and exponential temporal decay multiplier."""
        try:
            hd = datetime.fromisoformat(hist_date_str.replace("Z", "+00:00"))
            qd = datetime.fromisoformat(query_date_str.replace("Z", "+00:00"))
            delta_days = (qd - hd).days
            if delta_days < 0:
                delta_days = 0
                
            # Recency: 1.0 if today, asymptotically 0.0 for older
            recency = max(0.0, 1.0 - (delta_days / 365.0))
            
            # Decay: Halves every DECAY_HALF_LIFE_DAYS
            decay = math.pow(0.5, delta_days / self.config.DECAY_HALF_LIFE_DAYS)
            decay = max(self.config.MIN_DECAY_SCORE, decay)
            
            return recency, decay
        except Exception:
            return 0.5, 1.0

    def _cluster_deduplicate(self, candidates: List[Dict[str, Any]], embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Stage 5a: Clusters duplicate/redundant historical messages."""
        if len(candidates) <= 1:
            return candidates
            
        clustering = DBSCAN(eps=self.config.CLUSTER_EPS, min_samples=self.config.CLUSTER_MIN_SAMPLES, metric="cosine")
        labels = clustering.fit_predict(embeddings)
        
        unique_candidates = []
        seen_clusters = set()
        
        for idx, label in enumerate(labels):
            if label == -1 or label not in seen_clusters:
                unique_candidates.append(candidates[idx])
                if label != -1:
                    seen_clusters.add(label)
                    
        return unique_candidates

    def retrieve(self, msg_row: Dict[str, Any], context_features: Dict[str, Any]) -> Dict[str, Any]:
        """Core retrieval pipeline execution."""
        # Setup telemetry
        meta = {
            "candidate_count": 0,
            "filtered_count": 0,
            "bm25_score": 0.0,
            "embedding_score": 0.0,
            "recency_score": 0.0,
            "fusion_score": 0.0,
            "rerank_score": 0.0,
            "evidence_list": []
        }
        
        empty_retrieval = {
            "evidence_message_ids": ["none"],
            "retrieval_confidence": 0.0,
            "retrieval_reason": "No historical evidence found.",
            "retrieval_metadata": meta
        }
        
        # Stage 1: Candidate Retrieval & Caching BM25
        cache_key, all_user_candidates = self._get_candidates_unfiltered(msg_row)
        meta["candidate_count"] = 0
        
        if not all_user_candidates:
            return empty_retrieval

        query_raw = self._get_text_payload(msg_row)
        query_norm = QueryExpander.normalize(query_raw)
        query_date = str(msg_row.get("created_at", ""))
        
        # Stage 2: BM25 Scoring on UNFILTERED candidates
        if cache_key not in self.bm25_cache:
            unfiltered_texts = [self._get_text_payload(c) for c in all_user_candidates]
            unfiltered_norms = [QueryExpander.normalize(t) for t in unfiltered_texts]
            tokenized_corpus = [doc.split(" ") for doc in unfiltered_norms]
            self.bm25_cache[cache_key] = BM25Okapi(tokenized_corpus)
            
        bm25 = self.bm25_cache[cache_key]
        raw_bm25_scores = bm25.get_scores(query_norm.split(" "))
        
        # Now filter out the current message
        current_msg_id = str(msg_row.get("message_id", ""))
        ctype = msg_row.get("conversation_type", "personal")
        
        candidates = []
        bm25_scores = []
        for c, score in zip(all_user_candidates, raw_bm25_scores):
            if str(c.get("message_id")) == current_msg_id:
                continue
            if ctype == "business" and str(c.get("business_id", "")) != str(msg_row.get("business_id", "")):
                continue
            candidates.append(c)
            bm25_scores.append(score)
            
        meta["candidate_count"] = len(candidates)
        if not candidates:
            return empty_retrieval
            
        bm25_arr = np.array(bm25_scores)
        if len(bm25_arr) > 0 and max(bm25_arr) > 0:
            bm25_arr = bm25_arr / max(bm25_arr) # Normalize 0-1
            
        candidate_texts = [self._get_text_payload(c) for c in candidates]
        candidate_norms = [QueryExpander.normalize(t) for t in candidate_texts]
            
        # Stage 3: Dense Retrieval (Optimized Cosine Similarity)
        query_emb = self._get_embeddings_batched([query_norm])[0]
        cand_embs = self._get_embeddings_batched(candidate_norms)
        
        # Vectorized cosine similarity via NumPy
        cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
        cand_norms[cand_norms == 0] = 1.0
        normalized_cands = cand_embs / cand_norms
        
        q_norm = np.linalg.norm(query_emb)
        normalized_q = query_emb / (q_norm if q_norm > 0 else 1.0)
        
        aligned_faiss = np.dot(normalized_cands, normalized_q)

        # Stage 4: Fusion & Decay
        fusion_scores = []
        weights = self.config.get_weights()
        rel_score = context_features.get("group_priority_score", 0.5) if msg_row.get("conversation_type") == "group" else context_features.get("business_interaction_strength", 0.5)
        trust_score = (context_features.get("business_trust_score", 0.0) + 1.0) / 2.0 # Normalize -1..1 to 0..1
        
        for i, c in enumerate(candidates):
            recency, decay = self._compute_recency_and_decay(str(c.get("created_at", "")), query_date)
            
            raw_fusion = (
                (weights["bm25"] * bm25_arr[i]) +
                (weights["faiss"] * aligned_faiss[i]) +
                (weights["relationship"] * rel_score) +
                (weights["trust"] * trust_score) +
                (weights["recency"] * recency)
            )
            fusion_scores.append((raw_fusion, i, recency))
            
        fusion_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = fusion_scores[:10] # Top 10 for clustering
        
        top_k_candidates = [candidates[i] for _, i, _ in top_k]
        top_k_embs = np.array([cand_embs[i] for _, i, _ in top_k])
        
        # Stage 5: Clustering
        clustered_cands = self._cluster_deduplicate(top_k_candidates, top_k_embs)
        meta["filtered_count"] = len(candidates) - len(clustered_cands)
        
        if not clustered_cands:
            empty_retrieval["retrieval_reason"] = "All filtered"
            return empty_retrieval
            
        # Reranking (Cross-Encoder)
        use_ce = getattr(self.config, 'USE_CROSS_ENCODER', False)
        if use_ce and hasattr(self, 'cross_encoder'):
            rerank_pairs = [[query_norm, QueryExpander.normalize(self._get_text_payload(c))] for c in clustered_cands]
            rerank_scores = self.cross_encoder.predict(rerank_pairs)
            if not self.use_mocks:
                rerank_scores = 1 / (1 + np.exp(-rerank_scores))
            # Sort clustered_cands by rerank score
            sorted_by_rerank = sorted(zip(rerank_scores, clustered_cands), key=lambda x: x[0], reverse=True)
            best_cand = sorted_by_rerank[0][1]
            meta["rerank_score"] = float(sorted_by_rerank[0][0])
        else:
            best_cand = clustered_cands[0]
            meta["rerank_score"] = 0.0
            
        best_orig_idx = candidates.index(best_cand)
        best_score = float([fs for fs, i, _ in fusion_scores if i == best_orig_idx][0])
        
        meta["bm25_score"] = float(bm25_arr[best_orig_idx])
        meta["embedding_score"] = float(aligned_faiss[best_orig_idx])
        meta["fusion_score"] = best_score
        meta["recency_score"] = float([rs for _, i, rs in fusion_scores if i == best_orig_idx][0])
        
        # Populate structured evidence
        evidence_list = []
        evidence_list.append({
            "message_id": str(best_cand["message_id"]),
            "message_text": self._get_text_payload(best_cand),
            "similarity": float(aligned_faiss[best_orig_idx]),
            "retrieval_score": best_score
        })
        meta["evidence_list"] = evidence_list
        
        # Stage 6: Adaptive Thresholding
        threshold = self.config.BASE_CONFIDENCE_THRESHOLD
        if str(msg_row.get("conversation_type", "")) == "business":
            threshold += self.config.BUSINESS_THRESHOLD_MODIFIER
        if len(candidates) < 5:
            threshold += self.config.SPARSE_HISTORY_MODIFIER
            
        if best_score >= threshold:
            return {
                "evidence_message_ids": [str(best_cand["message_id"])],
                "retrieval_confidence": best_score,
                "retrieval_reason": f"High relevance semantic match. Score exceeded adaptive threshold {threshold:.2f}",
                "retrieval_metadata": meta
            }
        else:
            return {
                "evidence_message_ids": ["none"],
                "retrieval_confidence": best_score,
                "retrieval_reason": f"Best evidence below adaptive threshold {threshold:.2f}",
                "retrieval_metadata": meta
            }
