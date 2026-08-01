import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies import PolicyEngine
from decision_engine import DecisionEngine

def run_benchmark(iterations=10000):
    engine = DecisionEngine(PolicyEngine())
    payload = "Your verification code is 5678"
    context = {"sender_trust_score": 1.0, "interaction_strength": 0.8, "forwarded_count": 0, "group_muted": False}
    retrieval_meta = {"confidence": 0.9, "evidence": "h1"}
    
    start = time.time()
    for i in range(iterations):
        engine.process(f"msg_{i}", payload, context, retrieval_meta)
    
    return time.time() - start

if __name__ == "__main__":
    t = run_benchmark(10000)
    print("--- Decision Engine Runtime Benchmarks ---")
    print(f"10000 iterations: {t:.4f} seconds ({10000/t:,.0f} decisions/sec)")
