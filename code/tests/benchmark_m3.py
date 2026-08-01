import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multimodal_fusion import MultimodalFusionEngine

def run_benchmark(iterations=10000):
    text = "Please pay this urgent invoice of $500.00 by tomorrow. Call me at 555-123-4567."
    ocr_text = "Invoice #12345 Due tomorrow."
    asr_text = ""
    
    start = time.time()
    for _ in range(iterations):
        MultimodalFusionEngine.process(text, ocr_text, asr_text)
    
    return time.time() - start

if __name__ == "__main__":
    t = run_benchmark(10000)
    print("--- Multimodal Intelligence Benchmarks ---")
    print(f"10000 iterations: {t:.4f} seconds ({10000/t:,.0f} payloads/sec)")
