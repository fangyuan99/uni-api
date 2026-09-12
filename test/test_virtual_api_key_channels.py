import asyncio
from types import SimpleNamespace

from uni_api.api.alpha_search import _AlphaSearchExecution
from uni_api.runtime import (
    LingjingOpenapiHandler,
    MessagesPassthroughHandler,
    ResponsesRequestExecution,
    VideoTaskHandler,
)
from uni_api.routing.core import (
    VIRTUAL_API_KEY_INDEX,
    VIRTUAL_API_KEY_NAME,
    is_virtual_api_key_channel,
    virtual_api_key_index,
)


def _virtual_provider(index: int = 1, name: str = "sk-child") -> dict:
    return {
        "provider": name,
        "base_url": "",
        "model": [{"gpt-4.1": "gpt-4.1"}],
        VIRTUAL_API_KEY_INDEX: index,
        VIRTUAL_API_KEY_NAME: name,
    }


def test_virtual_channel_has_explicit_route_metadata():
    provider = _virtual_provider()
    assert is_virtual_api_key_channel(provider)
    assert virtual_api_key_index(provider) == 1


def test_responses_virtual_channel_dispatches_without_http(monkeypatch):
    calls = []

    async def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return "responses-result"

    execution = object.__new__(ResponsesRequestExecution)
    execution.handler = SimpleNamespace(request_responses=dispatch)
    execution.http_request = object()
    execution.request_data = object()
    execution.background_tasks = object()
    execution.endpoint = "/v1/responses"
    attempt = SimpleNamespace(provider=_virtual_provider(), state={"virtual_api_key_index": 1})

    result = asyncio.run(execution._execute_attempt(attempt))

    assert result == "responses-result"
    assert calls[0][1]["endpoint"] == "/v1/responses"
    assert calls[0][0][2] == 1


def test_messages_virtual_channel_dispatches_without_http(monkeypatch):
    calls = []
    handler = MessagesPassthroughHandler()

    async def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return "messages-result"

    monkeypatch.setattr(handler, "request_messages", dispatch)
    ctx = {
        "http_request": object(),
        "request_body": {"model": "gpt-4.1"},
        "background_tasks": object(),
        "endpoint": "/v1/messages",
    }
    attempt = SimpleNamespace(provider=_virtual_provider(), state={"virtual_api_key_index": 1})

    result = asyncio.run(handler._messages_execute_attempt(attempt, ctx))

    assert result == "messages-result"
    assert calls[0][1]["endpoint"] == "/v1/messages"
    assert calls[0][0][2] == 1


def test_video_virtual_channel_dispatches_without_http(monkeypatch):
    calls = []
    handler = VideoTaskHandler()

    async def dispatch(**kwargs):
        calls.append(kwargs)
        return "video-result"

    monkeypatch.setattr(handler, "_request_with_model_route", dispatch)
    ctx = {
        "http_request": object(),
        "request_body": {"model": "gpt-4.1"},
        "background_tasks": object(),
        "request_model_name": "gpt-4.1",
        "method": "POST",
        "task_id": None,
    }
    attempt = SimpleNamespace(provider=_virtual_provider(), state={"virtual_api_key_index": 1})

    result = asyncio.run(handler._video_execute_attempt(attempt, ctx))

    assert result == "video-result"
    assert calls[0]["api_index"] == 1


def test_alpha_search_virtual_channel_dispatches_without_http(monkeypatch):
    calls = []

    async def dispatch(**kwargs):
        calls.append(kwargs)
        return "search-result"

    handler = SimpleNamespace(request_search=dispatch)
    execution = object.__new__(_AlphaSearchExecution)
    execution.handler = handler
    execution.http_request = object()
    execution.request_body = {"model": "search", "query": "hello"}
    attempt = SimpleNamespace(provider=_virtual_provider(), state={"virtual_api_key_index": 1})

    result = asyncio.run(execution._execute_attempt(attempt))

    assert result == "search-result"
    assert calls[0]["api_index"] == 1


def test_lingjing_virtual_channel_dispatches_without_http(monkeypatch):
    calls = []
    handler = LingjingOpenapiHandler()

    async def dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return "lingjing-result"

    monkeypatch.setattr(handler, "request_openapi", dispatch)
    ctx = {
        "http_request": object(),
        "payload": {"model": "gpt-4.1"},
        "background_tasks": object(),
        "method_upper": "POST",
        "openapi_path": "/material/assets/create",
        "endpoint": "/v1/assets",
    }
    attempt = SimpleNamespace(provider=_virtual_provider(), state={"virtual_api_key_index": 1})

    result = asyncio.run(handler._lingjing_execute_attempt(attempt, ctx))

    assert result == "lingjing-result"
    assert calls[0][0][2] == 1
