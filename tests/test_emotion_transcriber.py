import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib
import pytest

from emotion_knowledge import EmotionTranscriptionPipeline, AudioTranscriber, TextEmotionAnnotator


class DummyTranscriber(AudioTranscriber):
    def __post_init__(self):
        self.called = False

    def __call__(self, audio_path: str) -> str:
        self.called = True
        return "hello world"


class DummyAnnotator(TextEmotionAnnotator):
    def __post_init__(self):
        self.called = False

    def __call__(self, text: str) -> str:
        self.called = True
        return "[happy] " + text


class TruncationCapturingAnnotator(TextEmotionAnnotator):
    def __post_init__(self):
        self.captured = {}

        def dummy(text, truncation=False, **kwargs):
            self.captured["truncation"] = truncation
            return [[{"label": "happy", "score": 1.0}]]

        self.pipeline = dummy


def test_pipeline_runs_with_dummies(tmp_path):
    audio_file = tmp_path / "fake.wav"
    audio_file.write_text("fake audio")
    pipeline = EmotionTranscriptionPipeline(DummyTranscriber(), DummyAnnotator())
    text, annotated = pipeline(str(audio_file))
    assert text == "hello world"
    assert annotated == "[happy] hello world"
    assert pipeline.transcriber.called
    assert pipeline.annotator.called


def test_annotator_truncates_long_text():
    annotator = TruncationCapturingAnnotator()
    long_text = "word " * 600
    result = annotator(long_text)
    assert annotator.captured.get("truncation") is True
    assert result.startswith("[happy]")

def test_transcriber_passes_pipeline_kwargs():
    """AudioTranscriber should initialize the HF pipeline with expected kwargs."""
    from unittest.mock import Mock, patch

    with patch("emotion_knowledge.pipeline", Mock(return_value=Mock())) as mock:
        AudioTranscriber()
        assert mock.call_args.args[0] == "automatic-speech-recognition"
        assert mock.call_args.kwargs.get("model") == "openai/whisper-base"
        assert mock.call_args.kwargs.get("language") == "de"
