from collections.abc import Iterable

# Answers of the vision model the way Bedrock returns them for a window of 6 frames, including the
# untidy ones (text around the JSON, impossible numbers), so the analyzer is tested with what really
# comes back and not only with clean JSON.
HEALTHY_GROUP = """{"guineaPigs": [
 {"color": "RED", "framesSeen": 6, "framesStill": 2, "feederVisits": 2, "waterVisits": 1,
  "avgGroupDistance": 0.3, "anomaly": 0.08, "confidence": 0.93},
 {"color": "BLUE", "framesSeen": 5, "framesStill": 1, "feederVisits": 1, "waterVisits": 0,
  "avgGroupDistance": 0.35, "anomaly": 0.1, "confidence": 0.9}]}"""

LETHARGIC_RED = """Here is the analysis:
{"guineaPigs": [
 {"color": "RED", "framesSeen": 6, "framesStill": 6, "feederVisits": 0, "waterVisits": 0,
  "avgGroupDistance": 0.85, "anomaly": 0.88, "confidence": 0.91},
 {"color": "BLUE", "framesSeen": 6, "framesStill": 2, "feederVisits": 1, "waterVisits": 1,
  "avgGroupDistance": 0.3, "anomaly": 0.07, "confidence": 0.92}]}"""

IMPOSSIBLE_NUMBERS = """{"guineaPigs": [
 {"color": "RED", "framesSeen": 9, "framesStill": 2, "anomaly": 0.1, "confidence": 0.9},
 {"color": "PINK", "framesSeen": 3, "framesStill": 1, "anomaly": 0.1, "confidence": 0.9},
 {"color": "BLUE", "framesSeen": 4, "framesStill": 5, "anomaly": 0.1, "confidence": 0.9}]}"""

NOT_JSON = "I can see two guinea pigs, but I am not sure about their marks."

NOBODY = '{"guineaPigs": []}'


class ScriptedRuntime:
    """A ModelRuntime that answers with the given texts, in order, and remembers what it was asked.
    It stands in for Bedrock so the real analyzer runs without AWS or cost."""

    def __init__(self, answers: Iterable[str]):
        self._answers = list(answers)
        self.calls: list[dict] = []

    async def converse(
        self, *, model_id: str, system: str, text: str, images: list[bytes], max_tokens: int = 800
    ) -> str:
        self.calls.append({"model_id": model_id, "text": text, "images": len(images)})
        if not self._answers:
            raise AssertionError("ScriptedRuntime ran out of answers")
        return self._answers.pop(0)
