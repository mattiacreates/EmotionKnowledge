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
        {"segment": 0, "speaker": "A", "start": 0.0, "end": 0.1, "text": "hi"},
        {"segment": 0, "speaker": "A", "start": 0.2, "end": 0.3, "text": "there"},
        {"segment": 0, "speaker": "A", "start": 0.35, "end": 0.45, "text": "everyone"},
        {"segment": 0, "speaker": "B", "start": 0.5, "end": 0.6, "text": "um"},
        {"segment": 0, "speaker": "B", "start": 0.65, "end": 0.75, "text": "yes"},
        {"segment": 0, "speaker": "B", "start": 0.8, "end": 0.9, "text": "indeed"},
        {"segment": 0, "speaker": "B", "start": 0.95, "end": 1.05, "text": "haha"},
        {"segment": 0, "speaker": "A", "start": 1.1, "end": 1.2, "text": "ok"},
        {"segment": 0, "speaker": "A", "start": 1.25, "end": 1.35, "text": "bye"},
    ]

    def raise_import_error(*args, **kwargs):
        raise ImportError("openpyxl missing")

    monkeypatch.setattr(pd.DataFrame, "to_excel", raise_import_error)
    out = tmp_path / "Single_Word_Transcript.xlsx"
    path = export_word_level_excel(
        words,
        str(out),
        multi_spk_seg_min_words=8,
        backchannel_run_min_words=3,
    )
    df = pd.read_csv(path)

    assert df.iloc[0:3]["is_backchannel"].all()
    assert df.iloc[3:7]["is_backchannel"].all()
    assert not df.iloc[7:]["is_backchannel"].any()

