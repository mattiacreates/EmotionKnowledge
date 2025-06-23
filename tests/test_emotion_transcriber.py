import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types

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


def test_pipeline_runs_with_dummies(tmp_path):
    audio_file = tmp_path / "fake.wav"
    audio_file.write_text("fake audio")
    pipeline = EmotionTranscriptionPipeline(DummyTranscriber(), DummyAnnotator())
    text, annotated = pipeline(str(audio_file))
    assert text == "hello world"
    assert annotated == "[happy] hello world"
    assert pipeline.transcriber.called
    assert pipeline.annotator.called
