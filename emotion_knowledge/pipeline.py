from langchain_core.runnables import RunnableSequence

from .audio_transcriber import AudioTranscriber
from .segment_db_writer import SegmentDBWriter
from .db_emotion_annotator import DBEmotionAnnotator
from .transcript_formatter import TranscriptFormatter


def emotion_transcription_pipeline(
    diarize: bool = False,
    model_size: str = "base",
    model_id: str = "oliverguhr/german-sentiment-bert",
    db_path: str = "db",
    clip_dir: str = "clips",
    load_in_8bit: bool = False,
    batch_size: int = 8,
):
    transcriber = AudioTranscriber(model_size=model_size, diarize=diarize)
    writer = SegmentDBWriter(db_path=db_path, clip_dir=clip_dir)
    annotator = DBEmotionAnnotator(
        db_path=db_path,
        model_name=model_id,
        batch_size=batch_size,
        load_in_8bit=load_in_8bit,
    )
    formatter = TranscriptFormatter()
    return RunnableSequence(transcriber) | writer | annotator | formatter
