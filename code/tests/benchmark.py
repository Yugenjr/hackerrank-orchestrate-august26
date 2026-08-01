import os
import sys
import time
import cProfile
import tracemalloc
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
from feature_engineering import ContextBuilder

def generate_fake_data(rows=10000):
    os.makedirs("dataset_bench", exist_ok=True)
    
    user_ids = [f"u{i}" for i in range(min(10000, rows))]
    pd.DataFrame({
        "user_id": user_ids,
        "do_not_disturb_window": ["22:00-07:00"] * len(user_ids),
        "notifications_dismissed_30d": [5] * len(user_ids),
        "messages_opened_30d": [15] * len(user_ids)
    }).to_csv("dataset_bench/users.csv", index=False)
    
    for f in ["groups.csv", "group_members.csv", "business_accounts.csv", 
              "user_business_history.csv", "daily_notification_summary.csv", 
              "message_events.csv", "messages.csv", "message_history.csv"]:
        if f != "users.csv":
            pd.DataFrame(columns=["id"]).to_csv(f"dataset_bench/{f}", index=False)

def run_benchmark(iterations=10000):
    loader = DataLoader("dataset_bench")
    loader.load_all()
    builder = ContextBuilder(loader)
    
    msg_row = {
        "user_id": "u1",
        "conversation_type": "personal",
        "created_at": "2026-08-01T12:00:00Z",
        "forwarded_count": 0
    }
    
    start_time = time.time()
    for _ in range(iterations):
        builder.build_context(msg_row)
    
    return time.time() - start_time

if __name__ == "__main__":
    generate_fake_data()
    
    print("--- Runtime Benchmarks ---")
    for size in [10_000, 100_000, 1_000_000]:
        t = run_benchmark(size)
        print(f"{size} iterations: {t:.4f} seconds ({size/t:,.0f} it/s)")
        
    print("\n--- Memory Allocation (1M) ---")
    tracemalloc.start()
    run_benchmark(1_000_000)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")
    tracemalloc.stop()
    
    print("\n--- cProfile (100k) ---")
    cProfile.run("run_benchmark(100000)", sort='tottime')
