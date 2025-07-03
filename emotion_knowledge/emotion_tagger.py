from typing import Dict, List
from transformers import pipeline

class MultimodalEmotionTagger:
    """Predict emotion from both text and audio input."""

    def __init__(
        self,
        text_model: str = "oliverguhr/german-sentiment-bert",
        # the original SpeechBrain model was removed from the hub. Use
        # a publicly available alternative for speech emotion recognition.
        audio_model: str = "superb/wav2vec2-base-superb-er",
    ) -> None:
        self.text_classifier = pipeline("text-classification", model=text_model)
        self.audio_classifier = pipeline("audio-classification", model=audio_model)

    def _prob_dict(self, outputs: List[Dict[str, float]]) -> Dict[str, float]:
        prob = {}
        for out in outputs:
            label = out.get("label")
            score = out.get("score", 0.0)
            prob[label] = prob.get(label, 0.0) + score
        return prob

    def invoke(self, text: str, audio_path: str) -> str:
        text_res = self.text_classifier(text)
        audio_res = self.audio_classifier(audio_path)
        t_probs = self._prob_dict(text_res)
        a_probs = self._prob_dict(audio_res)
        all_labels = set(t_probs) | set(a_probs)
        combo = {}
        for label in all_labels:
            combo[label] = t_probs.get(label, 0.0) + a_probs.get(label, 0.0)
        if not combo:
            return ""
        return max(combo, key=combo.get)


