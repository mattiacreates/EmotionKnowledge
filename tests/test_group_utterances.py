import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emotion_knowledge import _group_utterances


def test_grouping_updates_end_time():
    segments = [
        {"speaker": "speaker_01", "start": 0.0, "end_time": 0.5, "word": "Hallo"},
        {"speaker": "speaker_01", "start": 0.5, "end": 1.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(0.0)
    assert result[0]["end"] == pytest.approx(1.0)
    assert result[0]["text"] == "Hallo Welt"


def test_single_word_has_end_time():
    segments = [
        {"speaker": "speaker_02", "start": 1.5, "end_time": 2.0, "word": "Test"}
    ]
    result = _group_utterances(segments)
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(1.5)
    assert result[0]["end"] == pytest.approx(2.0)
    assert result[0]["text"] == "Test"


def test_zero_end_time_is_filled():
    segments = [
        {"speaker": "speaker_01", "start": 0.0, "end_time": 0.0, "word": "Hallo"},
        {"speaker": "speaker_01", "start": 0.5, "end": 1.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(0.0)
    assert result[0]["end"] == pytest.approx(1.0)
    assert result[0]["text"] == "Hallo Welt"

