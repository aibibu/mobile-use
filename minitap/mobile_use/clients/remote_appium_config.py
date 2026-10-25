"""Configuration for an existing iOS device on a GADS/Appium grid."""

import math
import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class RemoteAppiumClientConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    hub_url: str
    udid: str
    client_secret: SecretStr = Field(repr=False, exclude=True)
    timeout: float = Field(default=120, gt=0, le=300)
    queue_timeout: int = Field(default=10, ge=0, le=300)

    @field_validator("hub_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "%" in parsed.path
            or "\\" in value
            or any(part in {".", ".."} for part in parsed.path.split("/"))
            or any(ord(char) <= 32 for char in value)
        ):
            raise ValueError("hub_url must be an HTTP(S) grid URL without credentials or query")
        try:
            parsed.port
        except ValueError:
            raise ValueError("hub_url has an invalid port") from None
        return value.rstrip("/")

    @field_validator("udid")
    @classmethod
    def validate_udid(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", value):
            raise ValueError("An explicit valid device UDID is required")
        return value

    @field_validator("client_secret")
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("client_secret is required")
        return value

    @model_validator(mode="after")
    def validate_timeouts(self):
        if not math.isfinite(self.timeout) or self.queue_timeout >= self.timeout:
            raise ValueError("timeout must be finite and exceed queue_timeout")
        return self
