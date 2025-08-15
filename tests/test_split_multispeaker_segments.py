import os
import sys
import logging
import pytest

# Silence logs
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emotion_knowledge import _split_multispeaker_segments


def test_split_long_segment_with_interjection():
    words = [
        {"segment": 0, "speaker": "A", "start": 0.0, "end": 0.5, "word": "A1"},
        {"segment": 0, "speaker": "A", "start": 0.5, "end": 1.0, "word": "A2"},
        {"segment": 0, "speaker": "A", "start": 1.0, "end": 1.5, "word": "A3"},
        {"segment": 0, "speaker": "B", "start": 1.5, "end": 2.0, "word": "B1"},
        {"segment": 0, "speaker": "B", "start": 2.0, "end": 2.5, "word": "B2"},
        {"segment": 0, "speaker": "B", "start": 2.5, "end": 3.0, "word": "B3"},
        {"segment": 0, "speaker": "A", "start": 3.0, "end": 3.5, "word": "A4"},
        {"segment": 0, "speaker": "A", "start": 3.5, "end": 4.0, "word": "A5"},
    ]
    segments = _split_multispeaker_segments(words)
    assert [s["speaker"] for s in segments] == ["A", "B"]
    assert segments[0]["text"] == "A1 A2 A3"
    assert segments[1]["text"] == "B1 B2 B3 A4 A5"
    assert [w["segment"] for w in words] == [1, 1, 1, 2, 2, 2, 2, 2]


def test_initial_short_run_absorbed_and_short_segment_preserved():
    words = [
        {"segment": 0, "speaker": "X", "start": 0.0, "end": 0.2, "word": "hey"},
        {"segment": 0, "speaker": "X", "start": 0.2, "end": 0.4, "word": "there"},
        {"segment": 1, "speaker": "A", "start": 0.4, "end": 0.6, "word": "uh"},
        {"segment": 1, "speaker": "A", "start": 0.6, "end": 0.8, "word": "well"},
        {"segment": 1, "speaker": "B", "start": 0.8, "end": 1.0, "word": "I"},
        {"segment": 1, "speaker": "B", "start": 1.0, "end": 1.2, "word": "think"},
        {"segment": 1, "speaker": "B", "start": 1.2, "end": 1.4, "word": "so"},
        {"segment": 1, "speaker": "A", "start": 1.4, "end": 1.6, "word": "yes"},
        {"segment": 1, "speaker": "A", "start": 1.6, "end": 1.8, "word": "indeed"},
        {"segment": 1, "speaker": "A", "start": 1.8, "end": 2.0, "word": "sure"},
    ]
    segments = _split_multispeaker_segments(words)
    assert [s["speaker"] for s in segments] == ["X", "B", "A"]
    assert segments[0]["text"] == "hey there"
    assert segments[1]["text"] == "uh well I think so"
    assert segments[2]["text"] == "yes indeed sure"
    assert [w["segment"] for w in words] == [1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
