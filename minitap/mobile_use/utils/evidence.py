"""Private, per-observation evidence captured from the actual agent data path.

This directory is separate from legacy GIF inputs. No additional device reads
are performed here, and no model client objects or HTTP headers are persisted.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr

from minitap.mobile_use.context import MobileUseContext
from minitap.mobile_use.controllers.device_controller import ScreenDataResponse

_LOCK = threading.Lock()
_NAME = re.compile(r"[a-zA-Z0-9_-]+\Z")
_PRIVATE_KEYS = {
    "api_key",
    "apikey",
    "client_secret",
    "clientsecret",
    "authorization",
    "password",
    "access_token",
    "accesstoken",
    "refresh_token",
    "refreshtoken",
    "id_token",
    "idtoken",
}


def _private_key(key: Any) -> bool:
    name = str(key).lower().replace("-", "_")
    return name in _PRIVATE_KEYS or name.endswith("_api_key")


def _root(ctx: MobileUseContext) -> Path | None:
    setup = ctx.execution_setup
    if setup is None or setup.traces_path is None or not setup.trace_name:
        return None
    base = setup.traces_path.resolve()
    root = (base / setup.trace_name / "evidence").resolve()
    if not root.is_relative_to(base):
        raise ValueError("Evidence directory must be inside the trace directory")
    return root


def _safe(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return "[redacted]"
    if isinstance(value, BaseModel):
        return _safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if _private_key(key) else _safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def _mkdir(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _write(path: Path, data: bytes) -> None:
    _mkdir(path.parent)
    # Exclusive creation detects unexpected reuse instead of silently replacing evidence.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def _json(path: Path, data: Any) -> None:
    _write(
        path,
        (json.dumps(_safe(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(),
    )


def _append(path: Path, data: dict) -> None:
    _mkdir(path.parent)
    line = (json.dumps(_safe(data), ensure_ascii=False, allow_nan=False) + "\n").encode()
    with _LOCK:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write(line)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def write_run_metadata(trace_dir: Path, name: str, payload: Any) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid run metadata name")
    _json(trace_dir / "evidence" / f"{name}.json", payload)


async def begin_observation(ctx: MobileUseContext, screen: ScreenDataResponse) -> str | None:
    root = _root(ctx)
    if root is None:
        return None
    assert ctx.execution_setup is not None
    ctx.execution_setup.observation_count += 1
    observation_id = f"round-{ctx.execution_setup.observation_count:04d}"
    folder = root / observation_id
    image_bytes = base64.b64decode(screen.base64, validate=True)
    image_name = "screen.png" if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") else "screen.jpg"
    xml = screen.raw_xml

    def save() -> None:
        _write(folder / image_name, image_bytes)
        _json(folder / "elements.json", screen.elements)
        if xml is not None:
            _write(folder / "source.xml", xml.encode("utf-8"))
        _json(
            folder / "observation.json",
            {
                "observation_id": observation_id,
                "recorded_at": _now(),
                "width": screen.width,
                "height": screen.height,
                "platform": screen.platform,
                "screen": image_name,
                "screen_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "raw_xml_available": xml is not None,
                "source": "source.xml" if xml is not None else None,
                "element_count": len(screen.elements),
            },
        )

    await asyncio.to_thread(save)
    await record_event(ctx, observation_id, "observation", {"element_count": len(screen.elements)})
    return observation_id


async def record_model_input(
    ctx: MobileUseContext,
    observation_id: str | None,
    agent: str,
    messages: list,
    model: Any,
) -> None:
    root = _root(ctx)
    if root is None or observation_id is None:
        return
    if not _NAME.fullmatch(observation_id) or not _NAME.fullmatch(agent):
        raise ValueError("Invalid evidence identifier")
    folder = root / observation_id
    call_id = f"{agent}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"

    def save() -> None:
        image_index = 0

        def detach(value: Any) -> Any:
            nonlocal image_index
            if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
                header, encoded = value.split(";base64,", 1)
                raw = base64.b64decode(encoded, validate=True)
                image_index += 1
                ext = "png" if header == "data:image/png" else "jpg"
                filename = f"{call_id}-image-{image_index}.{ext}"
                _write(folder / filename, raw)
                return {
                    "artifact": filename,
                    "mime_type": header[5:],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            if isinstance(value, list):
                return [detach(item) for item in value]
            if isinstance(value, dict):
                return {key: detach(item) for key, item in value.items()}
            return value

        _json(
            folder / f"{call_id}-input.json",
            {
                "call_id": call_id,
                "observation_id": observation_id,
                "model": model,
                "format": "LangChain messages; inline image bytes stored as linked artifacts",
                "messages": detach(_safe(messages)),
            },
        )

    await asyncio.to_thread(save)
    await record_event(
        ctx,
        observation_id,
        f"{agent}_input",
        {"artifact": f"{call_id}-input.json", "call_id": call_id},
    )


async def record_event(
    ctx: MobileUseContext,
    observation_id: str | None,
    kind: str,
    payload: Any,
) -> None:
    root = _root(ctx)
    if root is None or observation_id is None:
        return
    if not _NAME.fullmatch(observation_id) or not _NAME.fullmatch(kind):
        raise ValueError("Invalid evidence identifier")
    event = {
        "recorded_at": _now(),
        "observation_id": observation_id,
        "kind": kind,
        "payload": payload,
    }
    await asyncio.to_thread(_append, root / observation_id / "events.jsonl", event)
