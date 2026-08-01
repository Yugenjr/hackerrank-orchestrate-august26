import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ner_engine import NEREngine
from structural_extractors import StructuralExtractor
from contradiction_engine import ContradictionEngine
from notification_signals import NotificationSignalEngine
from multimodal_fusion import MultimodalFusionEngine

def test_ner_engine():
    res = NEREngine.extract_entities("Call me at 555-123-4567 or visit https://test.com to pay $500.00.")
    assert len(res.phones) == 1
    assert res.phones[0].value == "555-123-4567"
    assert len(res.urls) == 1
    assert res.urls[0].value == "https://test.com"
    assert len(res.money) == 1
    assert res.money[0].value == "$500.00"

def test_structural_extractors():
    res = StructuralExtractor.extract("Please sign this document by tomorrow.")
    assert len(res.deadlines) == 1
    assert res.deadlines[0].value == "Relative Deadline"
    assert len(res.action_items) == 1
    assert res.action_items[0].value == "Imperative Action"

def test_contradiction():
    res = ContradictionEngine.detect(text="Pay this urgent invoice", ocr_text="Free gift inside")
    assert res.has_contradiction == True
    
    res2 = ContradictionEngine.detect(text="Hello", ocr_text="World")
    assert res2.has_contradiction == False

def test_notification_signals():
    res = NotificationSignalEngine.extract(text="Please pay the urgent invoice by tomorrow.")
    assert res.urgency.value == "high"
    assert res.requires_action.value == True
    assert res.payment_related.value == True
    assert res.business_related.value == True

def test_multimodal_fusion():
    ctx = MultimodalFusionEngine.process(text="Pay $50", ocr_text="Due tomorrow")
    assert ctx.has_media == True
    assert len(ctx.entities.money) == 1
    assert len(ctx.structures.deadlines) == 1
    assert ctx.features.payment_related.value == True
