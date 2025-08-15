import os

import pandas as pd
import pytest

from emotion_knowledge import export_word_level_excel


def test_export_word_level_excel_fallback_csv(tmp_path, monkeypatch):
    words = [
        {"segment": 0, "speaker": "A", "start": 0.0, "end": 0.4, "text": "hi"},
        {"segment": 1, "speaker": "B", "start": 0.5, "end": 0, "text": "there"},
    ]

    def raise_import_error(*args, **kwargs):
        raise ImportError("openpyxl missing")

    monkeypatch.setattr(pd.DataFrame, "to_excel", raise_import_error)
    out = tmp_path / "Single_Word_Transcript.xlsx"
    path = export_word_level_excel(words, str(out))

    assert path.endswith(".csv")
    assert os.path.exists(path)

    df = pd.read_csv(path)
    assert list(df.columns) == [
        "idx",
        "segment",
        "speaker",
        "start",
        "end",
        "duration",
        "text",
        "Is_backchannel",
    ]
    # missing end timestamp should fallback to start
    assert df.loc[1, "end"] == pytest.approx(df.loc[1, "start"])
    # duration column should be max(0, end-start)
    assert df.loc[0, "duration"] == pytest.approx(
        df.loc[0, "end"] - df.loc[0, "start"]
    )


def test_export_word_level_excel_backchannel_tags(tmp_path, monkeypatch):
    words = [
        {"segment": 0, "speaker": "A", "start": 0.0, "end": 0.1, "text": "Hi"},
        {"segment": 0, "speaker": "A", "start": 0.1, "end": 0.2, "text": "there"},
        {"segment": 0, "speaker": "B", "start": 0.2, "end": 0.3, "text": "can"},
        {"segment": 0, "speaker": "B", "start": 0.3, "end": 0.4, "text": "you"},
        {"segment": 0, "speaker": "B", "start": 0.4, "end": 0.5, "text": "hear"},
        {"segment": 0, "speaker": "C", "start": 0.5, "end": 0.6, "text": "me"},
        {"segment": 0, "speaker": "C", "start": 0.6, "end": 0.7, "text": "?"},
        {"segment": 0, "speaker": "A", "start": 0.7, "end": 0.8, "text": "yes"},
        {"segment": 1, "speaker": "D", "start": 0.0, "end": 0.1, "text": "Well"},
        {"segment": 1, "speaker": "E", "start": 0.1, "end": 0.2, "text": "I"},
        {"segment": 1, "speaker": "E", "start": 0.2, "end": 0.3, "text": "think"},
        {"segment": 1, "speaker": "E", "start": 0.3, "end": 0.4, "text": "so"},
        {"segment": 1, "speaker": "D", "start": 0.4, "end": 0.5, "text": "okay"},
    ]

    def raise_import_error(*args, **kwargs):
        raise ImportError("openpyxl missing")

    monkeypatch.setattr(pd.DataFrame, "to_excel", raise_import_error)
    out = tmp_path / "Single_Word_Transcript.xlsx"
    path = export_word_level_excel(words, str(out))
    df = pd.read_csv(path)

    # segment 0: speaker B has 3-word run -> backchannel
    assert df[(df["segment"] == 0) & (df["speaker"] == "B")]["Is_backchannel"].all()
    # speaker C run is only 2 words -> not backchannel
    assert not df[(df["segment"] == 0) & (df["speaker"] == "C")]["Is_backchannel"].any()
    # segment 1: speaker E has 3-word run -> backchannel
    assert df[(df["segment"] == 1) & (df["speaker"] == "E")]["Is_backchannel"].all()
    # primary speaker D remains False
    assert not df[(df["segment"] == 1) & (df["speaker"] == "D")]["Is_backchannel"].any()

