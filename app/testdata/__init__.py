"""Test data to exercise the models without a camera, a microphone or AWS.

- audio: synthetic WAV clips with a known expected label for the audio classifier.
- scenarios: deterministic behavior windows (healthy, lethargic, distress, recovery) to drive the
  backend's anomaly chain from NORMAL up to ALERT and back.
- model_answers: recorded answers of the vision model, replayed by ScriptedRuntime so the real
  Bedrock analyzer runs offline.

Nothing here is a recording of a real cage: every sample is generated, so no audio or frame
files end up in the repo.
"""
