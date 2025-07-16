import os
import sys
import logging
import pytest

# Silence package logs during tests
logging.disable(logging.CRITICAL)

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


def test_end_time_extended_to_next_start():
    segments = [
        {"speaker": "speaker_01", "start": 0.0, "end": 1.0, "word": "Hallo"},
        {"speaker": "speaker_01", "start": 5.0, "end": 5.5, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 2
    # end of first utterance should match start of second
    assert result[0]["end"] == pytest.approx(result[1]["start"])


def test_short_interjection_is_merged():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 0.6, "word": "hm"},
        {"speaker": "s1", "start": 0.6, "end": 1.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(0.0)
    assert result[0]["end"] == pytest.approx(1.0)
    assert result[0]["text"] == "Hallo hm Welt"


def test_long_single_word_interjection_is_merged():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 1.4, "word": "hm"},
        {"speaker": "s1", "start": 1.4, "end": 2.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(0.0)
    assert result[0]["end"] == pytest.approx(2.0)
    assert result[0]["text"] == "Hallo hm Welt"


def test_multi_word_interjection_over_one_second_not_merged():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {
            "speaker": "s2",
            "start": 0.5,
            "end": 1.7,
            "text": "ach so",
        },
        {"speaker": "s1", "start": 1.7, "end": 2.2, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 3
    assert result[0]["text"] == "Hallo"
    assert result[1]["text"] == "ach so"
    assert result[2]["text"] == "Welt"


def test_same_segment_id_overrides_gap():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo", "segment": 0},
        {"speaker": "s1", "start": 2.0, "end": 2.5, "word": "Welt", "segment": 0},
    ]
    result = _group_utterances(segments, max_gap=0.1)
    assert len(result) == 1
    assert result[0]["text"] == "Hallo Welt"

