import numpy as np
from typing import List, Dict

class EvaluationFramework:
    """
    Offline Evaluation Framework for the Retrieval Engine.
    Computes standard IR metrics: Recall@K, MRR, nDCG, Precision.
    """
    @staticmethod
    def _dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
        return 0.

    @staticmethod
    def ndcg_at_k(r, k):
        idcg = EvaluationFramework._dcg_at_k(sorted(r, reverse=True), k)
        if not idcg:
            return 0.
        return EvaluationFramework._dcg_at_k(r, k) / idcg

    @staticmethod
    def mrr(r):
        for index, item in enumerate(r):
            if item > 0:
                return 1.0 / (index + 1)
        return 0.0

    @staticmethod
    def evaluate(ground_truth: Dict[str, List[str]], predictions: Dict[str, List[str]]):
        metrics = {
            "Recall@1": 0.0,
            "Recall@5": 0.0,
            "MRR": 0.0,
            "nDCG": 0.0,
            "Precision": 0.0
        }
        
        n_queries = len(ground_truth)
        if n_queries == 0:
            return metrics
            
        for qid, true_ids in ground_truth.items():
            pred_ids = predictions.get(qid, [])
            
            # Binary relevance list
            relevance = [1 if pid in true_ids else 0 for pid in pred_ids]
            
            if len(pred_ids) > 0 and pred_ids[0] in true_ids:
                metrics["Recall@1"] += 1.0
                
            if any(p in true_ids for p in pred_ids[:5]):
                metrics["Recall@5"] += 1.0
                
            metrics["MRR"] += EvaluationFramework.mrr(relevance)
            metrics["nDCG"] += EvaluationFramework.ndcg_at_k(relevance, 5)
            
            if len(pred_ids) > 0:
                metrics["Precision"] += sum(relevance) / len(pred_ids)
                
        for k in metrics:
            metrics[k] /= n_queries
            
        return metrics
