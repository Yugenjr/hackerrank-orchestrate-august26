import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
from retrieval import RetrievalEngine

@pytest.fixture
def mock_loader(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    # 1. messages.csv
    pd.DataFrame({
        "message_id": ["m1", "m2"],
        "user_id": ["u1", "u1"],
        "conversation_type": ["personal", "business"],
        "business_id": ["", "b1"],
        "created_at": ["2026-08-01T10:00:00Z", "2026-08-01T12:00:00Z"],
        "message_text": ["hello", "your otp is 1234"],
        "ocr_text": ["", ""]
    }).to_csv(dataset_dir / "messages.csv", index=False)
    
    # 2. message_history.csv
    pd.DataFrame({
        "message_id": ["h1", "h2", "h3", "h4"],
        "user_id": ["u1", "u1", "u2", "u1"],
        "conversation_type": ["personal", "business", "personal", "business"],
        "business_id": ["", "b1", "", "b1"],
        "created_at": ["2026-07-01T10:00:00Z", "2026-07-20T10:00:00Z", "2026-06-01T10:00:00Z", "2026-01-01T10:00:00Z"],
        "message_text": ["hi there", "here is your otp 5678", "unrelated", "old otp 9999"],
        "ocr_text": ["", "", "", ""]
    }).to_csv(dataset_dir / "message_history.csv", index=False)
    
    # Empty files for the rest
    for f in ["users.csv", "groups.csv", "group_members.csv", "business_accounts.csv", "user_business_history.csv", "daily_notification_summary.csv", "message_events.csv"]:
        pd.DataFrame(columns=["id"]).to_csv(dataset_dir / f, index=False)
        
    loader = DataLoader(str(dataset_dir))
    loader.load_all()
    return loader

def test_retrieval_candidate_filtering(mock_loader):
    engine = RetrievalEngine(mock_loader, use_mocks=True)
    msg_row = {"message_id": "m2", "user_id": "u1", "conversation_type": "business", "business_id": "b1"}
    
    cands = engine._get_candidates(msg_row)
    assert len(cands) == 2  # h2, h4 (u1, b1)
    assert cands[0]["message_id"] == "h2"
    assert cands[1]["message_id"] == "h4"

def test_retrieval_recency_decay(mock_loader):
    engine = RetrievalEngine(mock_loader, use_mocks=True)
    # Query date: Aug 1
    # h2 is Jul 20 (12 days ago)
    # h4 is Jan 1 (212 days ago)
    
    r2, d2 = engine._compute_recency_and_decay("2026-07-20T10:00:00Z", "2026-08-01T10:00:00Z")
    r4, d4 = engine._compute_recency_and_decay("2026-01-01T10:00:00Z", "2026-08-01T10:00:00Z")
    
    assert r2 > r4  # More recent has higher recency score
    assert d2 > d4  # More recent has less decay (higher multiplier)
    
def test_retrieval_engine_pipeline(mock_loader):
    engine = RetrievalEngine(mock_loader, use_mocks=True)
    msg_row = {
        "message_id": "m2", 
        "user_id": "u1", 
        "conversation_type": "business", 
        "business_id": "b1",
        "created_at": "2026-08-01T12:00:00Z",
        "message_text": "your otp is 1234"
    }
    context = {"business_trust_score": 0.8, "business_interaction_strength": 0.9}
    
    result = engine.retrieve(msg_row, context)
    
    assert "evidence_message_ids" in result
    assert "retrieval_confidence" in result
    assert "retrieval_metadata" in result
    
    meta = result["retrieval_metadata"]
    assert meta["candidate_count"] == 2
    assert "bm25_score" in meta
    assert "recency_score" in meta
