import os
import sys
import time
import cProfile
import tracemalloc
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
from retrieval import RetrievalEngine

def generate_fake_data(rows=1000):
    os.makedirs("dataset_bench_m2", exist_ok=True)
    
    # 1 history file with N messages mapped to 1 user to test FAISS/BM25 latency
    pd.DataFrame({
        "message_id": [f"h{i}" for i in range(rows)],
        "user_id": ["u1"] * rows,
        "conversation_type": ["personal"] * rows,
        "business_id": [""] * rows,
        "created_at": ["2026-07-01T10:00:00Z"] * rows,
        "message_text": [f"random message text {i}" for i in range(rows)],
        "ocr_text": [""] * rows,
        "asr_transcript": [""] * rows
    }).to_csv("dataset_bench_m2/message_history.csv", index=False)
    
    for f in ["users.csv", "groups.csv", "group_members.csv", "business_accounts.csv", "user_business_history.csv", "daily_notification_summary.csv", "message_events.csv", "messages.csv"]:
        pd.DataFrame(columns=["id"]).to_csv(f"dataset_bench_m2/{f}", index=False)

def run_benchmark(iterations=1000):
    loader = DataLoader("dataset_bench_m2")
    loader.load_all()
    # use mocks to benchmark pure python/C++ (FAISS/BM25) latency without GPU/model overhead
    engine = RetrievalEngine(loader, use_mocks=True) 
    
    msg_row = {
        "message_id": "m_test",
        "user_id": "u1",
        "conversation_type": "personal",
        "created_at": "2026-08-01T12:00:00Z",
        "message_text": "random message"
    }
    context = {}
    
    start_time = time.time()
    for _ in range(iterations):
        engine.retrieve(msg_row, context)
        
    return time.time() - start_time

if __name__ == "__main__":
    generate_fake_data(1000) # 1000 historical messages for 1 user
    
    print("--- Retrieval Runtime Benchmarks (1000 history per user) ---")
    for size in [100, 1000]:
        t = run_benchmark(size)
        print(f"{size} iterations: {t:.4f} seconds ({size/t:,.0f} it/s)")
        
    print("\n--- Memory Allocation (100) ---")
    tracemalloc.start()
    run_benchmark(100)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")
    tracemalloc.stop()
