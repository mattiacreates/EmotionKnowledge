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


def test_end_time_preserved_by_default():
    segments = [
        {"speaker": "speaker_01", "start": 0.0, "end": 1.0, "word": "Hallo"},
        {"speaker": "speaker_01", "start": 5.0, "end": 5.5, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 2
    # first utterance keeps its original end
    assert result[0]["end"] == pytest.approx(1.0)


def test_end_time_not_modified_when_disabled():
    segments = [
        {"speaker": "speaker_01", "start": 0.0, "end": 1.0, "word": "Hallo"},
        {"speaker": "speaker_01", "start": 5.0, "end": 5.5, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 2
    # end timestamps come from the final word regardless of preserve_end_times
    assert result[0]["end"] == pytest.approx(1.0)


def test_short_interjection_not_backchannel():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 0.6, "word": "hm"},
        {"speaker": "s1", "start": 0.6, "end": 1.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert [utt["text"] for utt in result] == ["Hallo", "hm", "Welt"]
    assert "is_backchannel" not in result[1]


def test_long_single_word_interjection_not_backchannel():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 1.4, "word": "hm"},
        {"speaker": "s1", "start": 1.4, "end": 2.0, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert [utt["text"] for utt in result] == ["Hallo", "hm", "Welt"]
    assert "is_backchannel" not in result[1]


def test_multi_word_interjection_not_backchannel():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 1.7, "text": "ach so"},
        {"speaker": "s1", "start": 1.7, "end": 2.2, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert [utt["text"] for utt in result] == ["Hallo", "ach so", "Welt"]
    assert "is_backchannel" not in result[1]


def test_long_multi_word_interruption_not_backchannel():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {
            "speaker": "s2",
            "start": 0.5,
            "end": 2.0,
            "text": "das ist aber wirklich",
        },
        {"speaker": "s1", "start": 2.0, "end": 2.5, "word": "Welt"},
    ]
    result = _group_utterances(segments)
    assert [utt["text"] for utt in result] == ["Hallo", "das ist aber wirklich", "Welt"]
    assert "is_backchannel" not in result[1]


def test_run_based_backchannel_tagging():
    segments = [
        {"segment": 0, "speaker": "A", "start": 0.0, "end": 0.1, "text": "hi"},
        {"segment": 0, "speaker": "A", "start": 0.1, "end": 0.2, "text": "there"},
        {"segment": 0, "speaker": "A", "start": 0.2, "end": 0.3, "text": "everyone"},
        {"segment": 0, "speaker": "B", "start": 0.3, "end": 0.4, "text": "um"},
        {"segment": 0, "speaker": "B", "start": 0.4, "end": 0.5, "text": "yes"},
        {"segment": 0, "speaker": "B", "start": 0.5, "end": 0.6, "text": "indeed"},
        {"segment": 0, "speaker": "B", "start": 0.6, "end": 0.7, "text": "haha"},
        {"segment": 0, "speaker": "A", "start": 0.7, "end": 0.8, "text": "ok"},
        {"segment": 0, "speaker": "A", "start": 0.8, "end": 0.9, "text": "bye"},
    ]
    result = _group_utterances(
        segments,
        multi_spk_seg_min_words=8,
        backchannel_run_min_words=3,
    )
    assert len(result) == 3
    assert [utt["text"] for utt in result] == [
        "hi there everyone",
        "um yes indeed haha",
        "ok bye",
    ]
    assert result[0]["is_backchannel"] is True
    assert result[1]["is_backchannel"] is True
    assert "is_backchannel" not in result[2]


def test_merge_sentences_skips_backchannels():
    segments = [
        {"segment": 0, "speaker": "A", "start": 0.0, "end": 0.1, "text": "hi"},
        {"segment": 0, "speaker": "A", "start": 0.1, "end": 0.2, "text": "there"},

        {"segment": 0, "speaker": "B", "start": 0.2, "end": 0.3, "text": "um"},
        {"segment": 0, "speaker": "B", "start": 0.3, "end": 0.4, "text": "yes"},
        {"segment": 0, "speaker": "B", "start": 0.4, "end": 0.5, "text": "indeed"},
        {"segment": 0, "speaker": "A", "start": 0.5, "end": 0.6, "text": "ok"},
        {"segment": 1, "speaker": "A", "start": 0.7, "end": 0.8, "text": "bye"},
    ]
    result = _group_utterances(
        segments,
        merge_sentences=True,
        multi_spk_seg_min_words=5,
        backchannel_run_min_words=3,
    )
    assert len(result) == 3
    assert [utt["text"] for utt in result] == [
        "hi there",
        "um yes indeed",
        "ok bye",
    ]
    assert "is_backchannel" not in result[0]
    assert result[1]["is_backchannel"] is True
    assert "is_backchannel" not in result[2]


def test_same_segment_id_overrides_gap():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo", "segment": 0},
        {"speaker": "s1", "start": 2.0, "end": 2.5, "word": "Welt", "segment": 0},
    ]
    result = _group_utterances(segments, max_gap=0.1)
    assert len(result) == 1
    assert result[0]["text"] == "Hallo Welt"


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


def test_interjection_absorbed_when_requested():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "Hallo"},
        {"speaker": "s2", "start": 0.5, "end": 0.6, "word": "hm"},
        {"speaker": "s1", "start": 0.6, "end": 1.0, "word": "Welt"},
    ]
    result = _group_utterances(segments, absorb_interjections=True)
    assert len(result) == 1
    assert result[0]["text"] == "Hallo hm Welt"


def test_utterance_statistics_and_overlap():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "hi"},
        {"speaker": "s1", "start": 1.0, "end": 1.2, "word": "there"},
        {"speaker": "s2", "start": 1.1, "end": 1.5, "word": "yo"},
    ]
    result = _group_utterances(segments)
    assert len(result) == 2
    first, second = result
    assert first["n_words"] == 2
    assert first["duration"] == pytest.approx(1.2)
    assert first["words_per_sec"] == pytest.approx(2 / 1.2)
    assert first["mean_word_gap"] == pytest.approx(0.5)
    assert first["p95_word_gap"] == pytest.approx(0.5)
    assert first["overlaps_started"] is False
    assert second["overlaps_started"] is True
    assert second["n_words"] == 1
    assert second["mean_word_gap"] == pytest.approx(0.0)


def test_stats_with_segments_info():
    segments = [
        {"speaker": "s1", "start": 0.0, "end": 0.5, "word": "hello", "segment": 0},
        {"speaker": "s1", "start": 0.8, "end": 1.0, "word": "there", "segment": 0},
        {"speaker": "s2", "start": 1.5, "end": 2.0, "word": "hey", "segment": 1},
    ]
    result = _group_utterances(segments, segments_info=[{}, {}])
    assert len(result) == 2
    first, second = result
    assert first["n_words"] == 2
    assert first["duration"] == pytest.approx(1.0)
    assert first["words_per_sec"] == pytest.approx(2.0)
    assert first["mean_word_gap"] == pytest.approx(0.3)
    assert first["p95_word_gap"] == pytest.approx(0.3)
    assert second["n_words"] == 1
    assert second["duration"] == pytest.approx(0.5)

