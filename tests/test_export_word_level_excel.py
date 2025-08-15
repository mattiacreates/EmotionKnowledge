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
        "is_backchannel",
    ]
    # missing end timestamp should fallback to start
    assert df.loc[1, "end"] == pytest.approx(df.loc[1, "start"])
    # duration column should be max(0, end-start)
    assert df.loc[0, "duration"] == pytest.approx(
        df.loc[0, "end"] - df.loc[0, "start"]
    )


def test_export_word_level_excel_backchannel_tags(tmp_path, monkeypatch):
    words = [
        {"segment": 0, "speaker": "A", "start": 0.0, "end": 0.3, "text": "hi"},
        {"segment": 0, "speaker": "A", "start": 0.35, "end": 0.6, "text": "there"},
        {"segment": 0, "speaker": "A", "start": 0.65, "end": 0.9, "text": "everyone"},
        {"segment": 0, "speaker": "A", "start": 0.95, "end": 1.2, "text": "today"},
        {"segment": 1, "speaker": "B", "start": 1.3, "end": 1.4, "text": "um"},
        {"segment": 2, "speaker": "A", "start": 1.5, "end": 1.7, "text": "how"},
        {"segment": 2, "speaker": "A", "start": 1.75, "end": 1.95, "text": "are"},
        {"segment": 2, "speaker": "A", "start": 2.0, "end": 2.2, "text": "you"},
        {"segment": 2, "speaker": "A", "start": 2.25, "end": 2.6, "text": "doing"},
    ]

    def raise_import_error(*args, **kwargs):
        raise ImportError("openpyxl missing")

    monkeypatch.setattr(pd.DataFrame, "to_excel", raise_import_error)
    out = tmp_path / "Single_Word_Transcript.xlsx"
    path = export_word_level_excel(words, str(out))
    df = pd.read_csv(path)

    assert df[df["speaker"] == "B"]["is_backchannel"].all()
    assert not df[df["speaker"] == "A"]["is_backchannel"].any()

