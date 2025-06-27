from typing import Dict


class TranscriptFormatter:
    """Format annotated transcript segments."""

    def __init__(self):
        pass

    def __call__(self, transcript: Dict) -> str:
        lines = []
        for seg in transcript.get("segments", []):
            speaker = seg.get("speaker", "Speaker")
            emotion = seg.get("emotion") or seg.get("metadata", {}).get("emotion")
            if emotion is None:
                emotion = "?"
            lines.append(f"[{speaker}][{emotion}] {seg.get('text','').strip()}")
        return "\n".join(lines)
