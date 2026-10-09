import asyncio
import logging
from typing import Any, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class BedrockError(RuntimeError):
    """The model could not be reached or answered with an error."""


class ModelRuntime(Protocol):
    async def converse(
        self, *, model_id: str, system: str, text: str, images: list[bytes], max_tokens: int = 800
    ) -> str: ...


class BedrockRuntime:
    """Thin async wrapper over the Bedrock Converse API (one text prompt plus JPEG images).

    The boto3 client is created on first use, so building this object needs no credentials
    and no network (tests and the health check stay offline).
    """

    def __init__(self, region: str, client: Any | None = None):
        self._region = region
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            # short timeouts: a stuck call must not hold the window loop for minutes
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._region,
                config=Config(connect_timeout=5, read_timeout=45, retries={"max_attempts": 2}),
            )
        return self._client

    async def converse(
        self, *, model_id: str, system: str, text: str, images: list[bytes], max_tokens: int = 800
    ) -> str:
        content: list[dict[str, Any]] = [
            {"image": {"format": "jpeg", "source": {"bytes": image}}} for image in images
        ]
        content.append({"text": text})
        try:
            response = await asyncio.to_thread(
                self._get_client().converse,
                modelId=model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
            )
        except (BotoCoreError, ClientError) as exception:
            raise BedrockError(f"Bedrock call failed: {exception}") from exception
        parts = response.get("output", {}).get("message", {}).get("content", [])
        answer = "".join(part.get("text", "") for part in parts)
        if not answer:
            raise BedrockError("Bedrock answered without text")
        return answer
