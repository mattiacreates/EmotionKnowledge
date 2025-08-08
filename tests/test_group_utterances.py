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


def test_interjection_preserved():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 0.6, "word": "uh"},
        {"speaker": "s1", "start": 0.6, "end": 1.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert [utt["text"] for utt in result] == ["Hallo", "uh", "Welt"]


def test_interjection_merged_when_disabled():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 0.6, "word": "uh"},
        {"speaker": "s1", "start": 0.6, "end": 1.0, "word": "Welt"},
    ]
    result = _group_utterances(segments, keep_interjections=False)
    assert len(result) == 1
    assert result[0]["text"] == "Hallo uh Welt"


def test_segment_id_splits_on_speaker_change():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo", "segment": 0},
        {"speaker": "s2", "start": 0.5, "end": 0.6, "word": "uh", "segment": 0},
        {"speaker": "s1", "start": 0.6, "end": 1.0, "word": "Welt", "segment": 0},
    ]
    result = _group_utterances(segments)
    assert [utt["text"] for utt in result] == ["Hallo", "uh", "Welt"]


def test_end_time_not_truncated_with_overlap():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 1.5, "word": "Hallo"},
        {"speaker": "s2", "start": 1.0, "end": 2.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 2
    assert result[0]["end"] == pytest.approx(1.5)


def test_same_segment_split_when_gap_large():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo", "segment": 0},
        {"speaker": "s1", "start": 2.0, "end": 2.5, "word": "Welt", "segment": 0},
    ]
    result = _group_utterances(segments, max_gap=0.1)
    assert len(result) == 2
    assert [utt["text"] for utt in result] == ["Hallo", "Welt"]


def test_merge_sentences_combines_same_speaker():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 1.0, "word": "Hallo"},
        {"speaker": "s1", "start": 3.0, "end": 4.0, "word": "Welt"},
    ]
    result = _group_utterances(segments, merge_sentences=True)
    assert len(result) == 1
    assert result[0]["start"] == pytest.approx(0.0)
    assert result[0]["end"] == pytest.approx(4.0)
    assert result[0]["text"] == "Hallo Welt"

