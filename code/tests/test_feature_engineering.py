import os
import sys
import pytest
import pandas as pd
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from feature_engineering import ContextBuilder

@pytest.fixture
def mock_data_loader():
    loader = Mock()
    # Setup default returns
    loader.get_user.return_value = pd.Series({
        "do_not_disturb_window": "22:00-07:00",
        "notifications_dismissed_30d": 5,
        "messages_opened_30d": 15
    })
    loader.get_group_member.return_value = pd.Series({
        "group_muted_by_user": True,
        "replies_sent_30d": 2,
        "messages_read_30d": 10
    })
    loader.get_business.return_value = pd.Series({
        "verified": True,
        "official_domain": "example.com",
        "domain_used_by_sender": "example.com",
        "user_reports_30d": 5
    })
    loader.get_user_business_history.return_value = pd.Series({
        "promotions_opted_out_at": "2025-01-01T00:00:00Z",
        "messages_replied_30d": 1,
        "messages_opened_30d": 5
    })
    return loader

def test_is_dnd_active(mock_data_loader):
    builder = ContextBuilder(mock_data_loader)
    
    # 22:00 to 07:00
    assert builder._is_dnd_active("2026-08-01T23:00:00Z", "22:00-07:00")
    assert builder._is_dnd_active("2026-08-01T03:00:00Z", "22:00-07:00")
    assert not builder._is_dnd_active("2026-08-01T12:00:00Z", "22:00-07:00")
    
    # Missing or nan
    assert not builder._is_dnd_active("2026-08-01T23:00:00Z", "nan")
    
    # 09:00 to 17:00 (no wrap around)
    assert builder._is_dnd_active("2026-08-01T10:00:00Z", "09:00-17:00")
    assert not builder._is_dnd_active("2026-08-01T20:00:00Z", "09:00-17:00")

def test_build_context_group(mock_data_loader):
    builder = ContextBuilder(mock_data_loader)
    msg_row = pd.Series({
        "user_id": "u1",
        "conversation_type": "group",
        "group_id": "g1",
        "created_at": "2026-08-01T12:00:00Z",
        "forwarded_count": 0
    })
    
    context = builder.build_context(msg_row)
    assert context["conversation_type"] == "group"
    assert not context["user_dnd_active"]
    assert context["group_muted_by_user"]
    assert context["group_priority_score"] == 0.2  # 2 / 10
    assert context["user_global_dismissal_rate"] == 0.25  # 5 / (15 + 5)

def test_build_context_business(mock_data_loader):
    builder = ContextBuilder(mock_data_loader)
    msg_row = pd.Series({
        "user_id": "u1",
        "conversation_type": "business",
        "business_id": "b1",
        "created_at": "2026-08-01T23:00:00Z",
        "forwarded_count": 10
    })
    
    context = builder.build_context(msg_row)
    assert context["conversation_type"] == "business"
    assert context["user_dnd_active"]
    assert context["business_verified"]
    assert context["business_promotions_opted_out"]
    assert context["business_trust_score"] == 1.0  # +0.5 verified, +0.5 matching domain
    assert not context["baseline_scam_risk"]  # trust score > 0.5
