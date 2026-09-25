import enum
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Final, NoReturn, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.constants import (
    BEDROCK_INVOKE_SUPPORTED_THINKING_DISPLAY_VALUES,
    BEDROCK_INVOKE_UNSUPPORTED_MESSAGE_CONTENT_BLOCK_TYPES,
)
from litellm.litellm_core_utils.prompt_templates.factory import DEFAULT_USER_CONTINUE_MESSAGE

_KEY_MESSAGES: Final[str] = "messages"
_KEY_THINKING: Final[str] = "thinking"
_KEY_DISPLAY: Final[str] = "display"
_KEY_CONTENT: Final[str] = "content"
_KEY_TYPE: Final[str] = "type"
_KEY_TEXT: Final[str] = "text"
_KEY_OUTPUT_CONFIG: Final[str] = "output_config"


class OptInKnob(enum.Enum):
    DROP_PARAMS = "drop_params"
    MODIFY_PARAMS = "modify_params"


@dataclass(frozen=True, slots=True)
class OptIns:
    drop_params: bool
    modify_params: bool

    def allows(self, knob: OptInKnob) -> bool:
        match knob:
            case OptInKnob.DROP_PARAMS:
                return self.drop_params
            case OptInKnob.MODIFY_PARAMS:
                return self.modify_params
            case _:
                assert_never(knob)


@dataclass(frozen=True, slots=True)
class Offender:
    path: str
    knob: OptInKnob


@dataclass(frozen=True, slots=True)
class Unchanged:
    pass


@dataclass(frozen=True, slots=True)
class Sanitized:
    request: Mapping[str, JsonValue]
    removed: tuple[Offender, ...]


@dataclass(frozen=True, slots=True)
class Refused:
    offenders: tuple[Offender, ...]


SanitizeOutcome: TypeAlias = Unchanged | Sanitized | Refused


class _ContentBlockView(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    type: str


_CONTENT_BLOCK_UNION: TypeAlias = Annotated[_ContentBlockView | JsonValue, Field(union_mode="left_to_right")]

_PLACEHOLDER_TEXT_BLOCK: Final = _ContentBlockView.model_validate(
    types.MappingProxyType({_KEY_TYPE: "text", _KEY_TEXT: DEFAULT_USER_CONTINUE_MESSAGE[_KEY_CONTENT]})
)


class _MessageView(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    content: str | tuple[_CONTENT_BLOCK_UNION, ...] | None = None
    output_config: JsonValue | None = None


_MESSAGE_UNION: TypeAlias = Annotated[_MessageView | JsonValue, Field(union_mode="left_to_right")]


class _ThinkingView(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    display: str | None = None


class _RequestView(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    messages: tuple[_MESSAGE_UNION, ...] | None = None
    thinking: JsonValue | None = None


_REQUEST_VIEW_ADAPTER: Final = TypeAdapter(_RequestView)
_THINKING_VIEW_ADAPTER: Final = TypeAdapter(_ThinkingView)


def _thinking_view(raw_thinking: JsonValue | None) -> _ThinkingView | None:
    try:
        return _THINKING_VIEW_ADAPTER.validate_python(raw_thinking)
    except ValidationError:
        return None


def _unsupported_display(thinking: _ThinkingView | None) -> str | None:
    display: Final = thinking.display if thinking is not None else None
    return (
        display
        if isinstance(display, str) and display not in BEDROCK_INVOKE_SUPPORTED_THINKING_DISPLAY_VALUES
        else None
    )


def _message_param_offenders(message_index: int, message: _MessageView | JsonValue) -> tuple[Offender, ...]:
    if not isinstance(message, _MessageView) or _KEY_OUTPUT_CONFIG not in message.model_fields_set:
        return ()
    return (Offender(path=f"{_KEY_MESSAGES}[{message_index}].{_KEY_OUTPUT_CONFIG}", knob=OptInKnob.DROP_PARAMS),)


def _message_block_offenders(message_index: int, message: _MessageView | JsonValue) -> tuple[Offender, ...]:
    if not isinstance(message, _MessageView) or not isinstance(message.content, tuple):
        return ()
    return tuple(
        Offender(
            path=f"{_KEY_MESSAGES}[{message_index}].{_KEY_CONTENT}[{block_index}] ({_KEY_TYPE} '{block.type}')",
            knob=OptInKnob.MODIFY_PARAMS,
        )
        for block_index, block in enumerate(message.content)
        if isinstance(block, _ContentBlockView) and block.type in BEDROCK_INVOKE_UNSUPPORTED_MESSAGE_CONTENT_BLOCK_TYPES
    )


def _param_offenders(view: _RequestView, thinking: _ThinkingView | None) -> tuple[Offender, ...]:
    return tuple(
        offender
        for index, message in enumerate(view.messages or ())
        for offender in _message_param_offenders(index, message)
    ) + (
        (
            Offender(
                path=f"{_KEY_THINKING}.{_KEY_DISPLAY} (value '{_unsupported_display(thinking)}')",
                knob=OptInKnob.DROP_PARAMS,
            ),
        )
        if _unsupported_display(thinking) is not None
        else ()
    )


def _block_offenders(view: _RequestView) -> tuple[Offender, ...]:
    return tuple(
        offender
        for index, message in enumerate(view.messages or ())
        for offender in _message_block_offenders(index, message)
    )


def _filtered_content(
    content: tuple[_ContentBlockView | JsonValue, ...],
) -> tuple[_ContentBlockView | JsonValue, ...]:
    remaining: Final = tuple(
        block
        for block in content
        if not (
            isinstance(block, _ContentBlockView)
            and block.type in BEDROCK_INVOKE_UNSUPPORTED_MESSAGE_CONTENT_BLOCK_TYPES
        )
    )
    if remaining or not content:
        return remaining
    return (_PLACEHOLDER_TEXT_BLOCK,)


def _sanitize_message_view(message: _MessageView | JsonValue, opt_ins: OptIns) -> JsonValue:
    if not isinstance(message, _MessageView):
        return message
    updated: Final = (
        message.model_copy(update=types.MappingProxyType({_KEY_CONTENT: _filtered_content(message.content)}))
        if opt_ins.modify_params and isinstance(message.content, tuple)
        else message
    )
    excludes: Final = types.MappingProxyType(dict.fromkeys((str(_KEY_OUTPUT_CONFIG),), True))
    return updated.model_dump(mode="json", exclude_unset=True, exclude=excludes if opt_ins.drop_params else None)


def sanitize_for_bedrock_invoke(request: Mapping[str, object], opt_ins: OptIns) -> SanitizeOutcome:
    try:
        view: Final = _REQUEST_VIEW_ADAPTER.validate_python(request)
    except ValidationError:
        return Unchanged()

    thinking_view: Final = _thinking_view(view.thinking)
    offenders: Final = _param_offenders(view, thinking_view) + _block_offenders(view)
    blocked: Final = tuple(offender for offender in offenders if not opt_ins.allows(offender.knob))
    if blocked:
        return Refused(offenders=blocked)
    if not offenders:
        return Unchanged()

    sanitized_messages: Final = (
        tuple(_sanitize_message_view(message, opt_ins) for message in view.messages)
        if view.messages is not None
        else None
    )
    drop_display: Final = types.MappingProxyType(dict.fromkeys((str(_KEY_DISPLAY),), True))
    sanitized_thinking: Final = (
        thinking_view.model_dump(mode="json", exclude_unset=True, exclude=drop_display)
        if opt_ins.drop_params and thinking_view is not None and _unsupported_display(thinking_view) is not None
        else None
    )
    update: Final = types.MappingProxyType(
        {
            k: v
            for k, v in (
                (_KEY_MESSAGES, sanitized_messages),
                (_KEY_THINKING, sanitized_thinking),
            )
            if v is not None
        }
    )
    sanitized_request: Final = view.model_copy(update=update).model_dump(mode="json", exclude_unset=True)
    return Sanitized(request=sanitized_request, removed=offenders)


def raise_refusal(refused: Refused, model: str) -> NoReturn:
    param_offenders: Final = tuple(
        offender.path for offender in refused.offenders if offender.knob is OptInKnob.DROP_PARAMS
    )
    block_offenders: Final = tuple(
        offender.path for offender in refused.offenders if offender.knob is OptInKnob.MODIFY_PARAMS
    )
    param_error: Final = (
        f"Bedrock Invoke does not accept {', '.join(param_offenders)}. "
        "Set `drop_params: true` in this deployment's `litellm_params` (or `litellm_settings.drop_params: true` "
        "on the proxy, `litellm.drop_params = True` in the SDK) to have LiteLLM drop them (an unsupported "
        "thinking.display falls back to the model default), or remove them from the request."
    )
    block_error: Final = (
        f"Bedrock Invoke does not accept {', '.join(block_offenders)}. "
        "Set `litellm_settings.modify_params: true` on the proxy or `litellm.modify_params = True` "
        "in the SDK to have LiteLLM remove those blocks (a message left empty gets the placeholder "
        f"text '{DEFAULT_USER_CONTINUE_MESSAGE[_KEY_CONTENT]}'), or remove them from the request."
    )
    match (bool(param_offenders), bool(block_offenders)):
        case (True, True):
            raise litellm.UnsupportedParamsError(
                message=f"{param_error} {block_error}",
                model=model,
                llm_provider="bedrock",
            )
        case (True, False):
            raise litellm.UnsupportedParamsError(
                message=param_error,
                model=model,
                llm_provider="bedrock",
            )
        case (False, True):
            raise litellm.BadRequestError(
                message=block_error,
                model=model,
                llm_provider="bedrock",
            )
        case (False, False):
            raise AssertionError("Refused always carries at least one offender")
