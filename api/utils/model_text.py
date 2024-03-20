from dataclasses import dataclass
from transformers import pipeline

@dataclass
class SentimentPrediction:
    """Class representing a sentiment prediction result."""

    label: str
    score: float


def load_model_text():
    """Load a pre-trained sentiment analysis model.

    Returns:
        model (function): A function that takes a text input and returns a SentimentPrediction object.
    """
    model_hf = pipeline('sentiment-analysis', model='cointegrated/rubert-tiny-sentiment-balanced', device=-1)

    def model(text: str) -> SentimentPrediction:
        pred = model_hf(text)
        pred_best_class = pred[0]
        return SentimentPrediction(
            label=pred_best_class["label"],
            score=pred_best_class["score"],
        )

    return model
