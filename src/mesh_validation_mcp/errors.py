"""Structured tool errors.

Every error raised by a tool carries a machine-readable JSON envelope as its
string form, so the MCP client (an agent) can parse ``{"code", "message",
"hint"}`` out of the ``isError`` result and act on it.
"""

from __future__ import annotations

import json
from enum import Enum


class ErrorCode(str, Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    NOT_ABSOLUTE_PATH = "NOT_ABSOLUTE_PATH"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    LOAD_FAILED = "LOAD_FAILED"
    EMPTY_MESH = "EMPTY_MESH"
    MESH_TOO_LARGE = "MESH_TOO_LARGE"
    INVALID_EXPECTATION = "INVALID_EXPECTATION"
    INVALID_VIEW = "INVALID_VIEW"
    RENDER_FAILED = "RENDER_FAILED"


class MeshToolError(Exception):
    def __init__(self, code: ErrorCode, message: str, hint: str | None = None) -> None:
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(self.to_json())

    def to_json(self) -> str:
        payload: dict[str, str] = {"code": self.code.value, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        return json.dumps(payload)
