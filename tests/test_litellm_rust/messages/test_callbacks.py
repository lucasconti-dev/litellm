from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    request_body,
    request_headers,
)

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


@pytest.fixture(autouse=True)
def opt_messages_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def assert_served_natively(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.asyncio
@pytest.mark.parametrize("rollout", [Rollout.PYTHON_ONLY, Rollout.RUST_OPT_IN], ids=["python", "rust"])
async def test_messages_custom_logger_pre_request_hook_rewrites_tools(
    messages_server: RecordingServer, rollout: Rollout
) -> None:
    seen: Final = []

    class RewriteTool(CustomLogger):
        async def async_pre_request_hook(self, model, messages, kwargs):
            seen.append((model, kwargs["tools"][0]["name"]))
            return {**kwargs, "tools": [{**kwargs["tools"][0], "name": "renamed_tool"}]}

    litellm.callbacks.append(RewriteTool())
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, rollout), *catalog.RULES)):
        await litellm.anthropic.messages.acreate(
            **arguments(
                messages_server,
                tools=[{"name": "original_tool", "description": "Lookup", "input_schema": {"type": "object"}}],
            )
        )

    assert seen == [(MESSAGES_MODEL, "original_tool")]
    assert messages_server.requests[0].body["tools"][0]["name"] == "renamed_tool"


@pytest.mark.asyncio
async def test_native_messages_callbacks_see_the_provider_request_and_the_public_response(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], litellm_call_id="messages-success")
    )

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    sent: Final = messages_server.requests[0]
    assert sent.path == "/v1/messages"
    assert sent.body == {"model": "claude-sonnet-5", "messages": list(MESSAGES), "max_tokens": 64, "stream": False}
    pre_call: Final = recorder.wait_for("log_pre_api_call")
    assert request_body(pre_call[0].kwargs) == sent.body
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].call_type == "anthropic_messages"
    assert success[0].kwargs["litellm_call_id"] == "messages-success"
    assert success[0].response.choices[0].message.content == "Hello from native Messages"


@pytest.mark.asyncio
@pytest.mark.parametrize("raise_after_edit", [False, True], ids=["callback-returns", "callback-raises"])
async def test_native_messages_pre_call_logger_edits_reach_next_logger_and_provider(
    messages_server: RecordingServer, raise_after_edit: bool
) -> None:
    observed: Final = []

    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["temperature"] = 0.25
            request_headers(kwargs)["x-audit-tag"] = "reviewed"
            if raise_after_edit:
                raise RuntimeError("pre-call callback failed")

    class Observe(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            observed.append((request_body(kwargs)["temperature"], request_headers(kwargs)["x-audit-tag"]))

    await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Edit(), Observe()]))

    assert observed == [(0.25, "reviewed")]
    assert messages_server.requests[0].body["temperature"] == 0.25
    assert messages_server.requests[0].headers["x-audit-tag"] == "reviewed"


@pytest.mark.asyncio
async def test_native_messages_provider_error_reaches_caller_and_failure_callbacks_as_one_public_error(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(
        ResponseSpec(body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, status=400)
    )
    observed: Final = []

    class Observe(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("sync", kwargs["exception"]))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("async", kwargs["exception"]))

    with pytest.raises(litellm.BadRequestError) as raised:
        await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Observe()]))

    assert_served_natively(messages_server)
    assert [phase for phase, _ in observed] == ["sync", "async"]
    assert all(error is raised.value for _, error in observed)


@pytest.mark.asyncio
async def test_native_messages_stream_provider_error_fails_before_returning_an_iterator(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(
        ResponseSpec(
            body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad stream"}}, status=400
        )
    )
    recorder: Final = RecordingLogger()

    with pytest.raises(litellm.BadRequestError) as raised:
        await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True, callbacks=[recorder]))

    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is True
    assert recorder.wait_for("log_failure_event")[0].kwargs["exception"] is raised.value
    assert (await recorder.wait_for_async("async_log_failure_event"))[0].kwargs["exception"] is raised.value
    assert "async_log_success_event" not in recorder.names


def sse_payload() -> bytes:
    return b"".join(STREAM.payloads())


@pytest.mark.asyncio
async def test_native_messages_stream_relays_provider_events_and_logs_success_once_after_the_last_chunk(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    assert get_hidden_params_dict(stream) == {"additional_headers": {"x-litellm-rust": "true"}}
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [chunk async for chunk in stream]

    assert first + b"".join(rest) == sse_payload()
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is True
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].kwargs["stream"] is True
    assert success[0].kwargs["completion_start_time"] is not None
    assert success[0].response.choices[0].message.content == "Hello from native Messages"
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_messages_stream_closed_early_logs_success_once_for_what_was_delivered(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS, inter_payload_delay=0.05))
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    first: Final = await anext(stream)
    await stream.aclose()

    assert first == STREAM.payloads()[0]
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].response.choices[0].message.content == ""
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_native_messages_stream_disconnect_after_a_chunk_reports_failure_without_success(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(
        ResponseSpec(
            body=None,
            events=MESSAGES_EVENTS[:2],
            disconnect_after_payloads=True,
            inter_payload_delay=0.05,
        )
    )
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    first: Final = await anext(stream)
    with pytest.raises(litellm.APIConnectionError) as raised:
        async for _ in stream:
            pass

    assert first.startswith(b"event: message_start\n")
    assert_served_natively(messages_server)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    assert len(recorder.wait_for("log_pre_api_call")) == 1
    failure: Final = await recorder.wait_for_async("async_log_failure_event")
    assert len(failure) == 1
    assert failure[0].kwargs["exception"] is raised.value


def test_native_sync_messages_stream_relays_provider_events_and_logs_success_once(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = litellm.anthropic.messages.create(**arguments(messages_server, stream=True, callbacks=[recorder]))
    assert isinstance(stream, Iterator)
    assert get_hidden_params_dict(stream) == {"additional_headers": {"x-litellm-rust": "true"}}

    assert b"".join(stream) == sse_payload()
    assert_served_natively(messages_server)
    assert len(recorder.wait_for("async_log_success_event")) == 1


def test_native_sync_messages_stream_disconnect_reports_failure_without_success(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(
        ResponseSpec(body=None, events=MESSAGES_EVENTS[:2], disconnect_after_payloads=True, inter_payload_delay=0.05)
    )
    recorder: Final = RecordingLogger()

    stream: Final = litellm.anthropic.messages.create(**arguments(messages_server, stream=True, callbacks=[recorder]))
    first: Final = next(stream)
    with pytest.raises(litellm.APIConnectionError) as raised:
        tuple(stream)

    assert first.startswith(b"event: message_start\n")
    assert_served_natively(messages_server)
    failure: Final = recorder.wait_for("log_failure_event")
    assert len(failure) == 1
    assert failure[0].kwargs["exception"] is raised.value
    assert "async_log_success_event" not in recorder.names


def test_native_sync_messages_returns_the_provider_message(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    response: Final = litellm.anthropic.messages.create(**arguments(messages_server, callbacks=[recorder]))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert len(recorder.wait_for("log_success_event")) == 1


def test_native_messages_dispatches_each_callback_phase_once_when_logger_is_registered_multiple_times(
    messages_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()

    litellm.anthropic.messages.create(
        **arguments(
            messages_server,
            callbacks=[recorder, recorder],
            success_callback=[recorder],
            failure_callback=[recorder],
        )
    )
    recorder.wait_for("log_success_event")

    assert recorder.names.count("log_pre_api_call") == 1
    assert recorder.names.count("logging_hook") == 1
    assert recorder.names.count("log_success_event") == 1
    assert "log_failure_event" not in recorder.names
