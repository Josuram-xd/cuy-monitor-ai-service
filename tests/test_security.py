from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings, get_settings
from app.security import require_api_key


def test_api_key_dependency_rejects_missing_or_invalid_keys():
    test_app = FastAPI()
    test_app.dependency_overrides[get_settings] = lambda: Settings(
        api_key=SecretStr("expected-key"),
        _env_file=None,
    )

    @test_app.get("/protected", dependencies=[Depends(require_api_key)])
    def protected_route():
        return {"status": "ok"}

    client = TestClient(test_app)

    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "wrong-key"}).status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "expected-key"}).status_code == 200
