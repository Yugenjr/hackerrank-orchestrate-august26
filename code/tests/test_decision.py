import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies import PolicyEngine
from decision_engine import DecisionEngine
from reason_generator import ReasonGenerator
from schemas import FinalDecisionOutput

def test_reason_generator_validation():
    # Hallucination trap
    reason = ReasonGenerator.generate_and_validate("notify", "otp", None, "[Insert reason here]")
    assert "[Insert" not in reason
    assert reason == "Notify: Urgent authentication/OTP code."
    
    # Length validation
    long_reason = "a" * 200
    r2 = ReasonGenerator.generate_and_validate("notify", "personal", None, long_reason)
    assert len(r2) <= 150
    assert r2.endswith("...")

def test_policy_engine():
    engine = PolicyEngine()
    # Test Family Emergency
    ctx = {"interaction_strength": 0.8}
    res = engine.evaluate("Help me I am in the hospital", ctx)
    assert res is not None
    assert res[0] == "notify"
    assert res[1] == "urgent"
    
    # Test OTP
    res2 = engine.evaluate("Your verification code is 123456", {})
    assert res2 is not None
    assert res2[0] == "notify"
    
    # Test Muted Group
    res3 = engine.evaluate("Hey guys", {"group_muted": True})
    assert res3 is not None
    assert res3[0] == "mute"

def test_decision_engine_pipeline():
    engine = DecisionEngine(PolicyEngine())
    
    # Test short-circuit on OTP
    payload = "Here is your 123456 code"
    res, trace = engine.process("msg1", payload, {}, {"evidence": "msg_prev"})
    
    assert res.action == "notify"
    assert res.confidence == 1.0
    assert trace["steps"][0]["matched"] == True
    
    # Test fallback to LLM
    payload2 = "Just a normal chat"
    res2, trace2 = engine.process("msg2", payload2, {}, {"evidence": "none"})
    # Since Safety mocked to True and LLM mocked to digest
    assert res2.action == "digest"
    # Expected Conf: 0.5*0.8 (LLM) + 0.3*0.0 (Retrieval) + 0.2*0.5 (Trust) = 0.5
    assert res2.confidence == 0.5

def test_counterfactuals():
    engine = DecisionEngine(PolicyEngine())
    res = engine.counterfactual_analysis("msg3", "Hello", {"sender_trust_score": 1.0, "group_muted": False}, {})
    assert "baseline_action" in res
    assert "if_group_muted" in res
    assert res["if_group_muted"] == "mute"
