"""Emotion detection module powered by Watson NLP."""

import requests


WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_MODEL_HEADER = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyse):
    """Analyze text and return emotion scores with dominant emotion."""
    payload = {"raw_document": {"text": text_to_analyse}}
    response = requests.post(
        WATSON_EMOTION_URL,
        json=payload,
        headers=WATSON_MODEL_HEADER,
        timeout=30,
    )

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response_json = response.json()
    emotions = response_json["emotionPredictions"][0]["emotion"]
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }

