import argparse

from .pipeline import emotion_transcription_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="German speech-to-emotion transcription pipeline"
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--diarize", action="store_true", help="Use speaker diarization")
    parser.add_argument("--model-size", default="base", help="Whisper model size")
    parser.add_argument(
        "--emotion-model",
        default="oliverguhr/german-sentiment-bert",
        help="HuggingFace emotion model",
    )
    parser.add_argument("--db-path", default="db", help="ChromaDB directory")
    parser.add_argument("--clip-dir", default="clips", help="Directory for audio clips")
    parser.add_argument("--batch-size", type=int, default=8, help="Emotion batch size")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load emotion model in 8bit")
    args = parser.parse_args()

    pipeline = emotion_transcription_pipeline(
        diarize=args.diarize,
        model_size=args.model_size,
        model_id=args.emotion_model,
        db_path=args.db_path,
        clip_dir=args.clip_dir,
        batch_size=args.batch_size,
        load_in_8bit=args.load_in_8bit,
    )
    output = pipeline.invoke(args.audio)
    print(output)


if __name__ == "__main__":
    main()
