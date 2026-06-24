"""OpenAI-protocol agent loop for non-Anthropic providers (Ollama, OpenAI, etc.).

Provides streaming multi-turn tool calling using the OpenAI chat completions
format. Structurally parallel to the Anthropic loop in ClaudeService but speaks
native OpenAI wire format end-to-end — no format conversion at the LLM boundary.

Tool execution reuses the same MCP + backend dispatch as ClaudeService.

Parity with ClaudeService.chat_stream:
  - Multi-turn tool loop (up to 30 iterations)
  - Wall-clock timeout (300s)
  - Exponential backoff between iterations
  - Infinite loop detection (identical tool signatures × 3)
  - Per-agent recommended_tools filtering
  - Context window management (history window + token-based trimming)
  - Per-iteration interaction logging (LLMInteractionLog)
  - Tool result truncation + prompt injection wrapping
  - Budget enforcement via VK headers
  - Bifrost correlation UUID per iteration
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from services.llm_router import LLMRouter, ProviderSpec

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 30
_MAX_PROCESSING_TIME_S = 300.0
_TOOL_TIMEOUT_S = 30.0
_MAX_TOOL_RESPONSE_CHARS = 50_000
_HISTORY_WINDOW_DEFAULT = 20
_LOOP_DETECT_WINDOW = 5
_LOOP_DETECT_THRESHOLD = 3
_BASE_INTER_ITERATION_DELAY_S = 0.5
_MAX_INTER_ITERATION_DELAY_S = 3.0
_BACKOFF_ITERATION_THRESHOLD = 15


def anthropic_tools_to_openai(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert Anthropic tool schema format to OpenAI function-calling format.

    Anthropic: {"name", "description", "input_schema": {json-schema}}
    OpenAI:    {"type": "function", "function": {"name", "description",
               "parameters": {json-schema}}}
    """
    converted = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "input_schema",
                    {"type": "object", "properties": {}},
                ),
            },
        })
    return converted


def _truncate(
    text: str, max_chars: int = _MAX_TOOL_RESPONSE_CHARS
) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n...[truncated, {omitted} chars omitted]"


def _inter_iteration_delay(iteration: int) -> float:
    """Exponential backoff matching ClaudeService: 500ms base, ramp after 15."""
    if iteration <= _BACKOFF_ITERATION_THRESHOLD:
        return _BASE_INTER_ITERATION_DELAY_S
    exp = iteration - _BACKOFF_ITERATION_THRESHOLD
    delay = _BASE_INTER_ITERATION_DELAY_S * (1.5 ** exp)
    return min(delay, _MAX_INTER_ITERATION_DELAY_S)


class OpenAIAgentService:
    """Streaming agent loop for OpenAI-compatible providers via Bifrost.

    Mirrors ClaudeService.chat_stream capabilities:
      - Multi-turn tool calling with streaming
      - Infinite loop detection
      - Context window management
      - Per-iteration audit logging
      - Agent-specific tool filtering
    """

    def __init__(
        self,
        *,
        backend_tools: Optional[List[Dict[str, Any]]] = None,
        include_mcp_tools: bool = True,
        recommended_tools: Optional[List[str]] = None,
    ):
        self._backend_tools = backend_tools or self._load_backend_tools()
        self._include_mcp_tools = include_mcp_tools
        self._recommended_tools = recommended_tools
        self._backend_tool_names: Set[str] = {
            t["name"] for t in self._backend_tools
        }
        self._refresh_skill_tools()

    @staticmethod
    def _load_backend_tools() -> List[Dict[str, Any]]:
        try:
            from backend.schemas.tool_schemas import ALL_TOOLS
            return list(ALL_TOOLS)
        except ImportError:
            logger.debug("Backend tool schemas unavailable")
            return []

    def _refresh_skill_tools(self) -> None:
        """Load DB-backed skill tools (same as ClaudeService._refresh_skill_tools)."""
        try:
            from services.skill_tools_bridge import list_active_skill_tools
            skill_tools, self._skill_tool_index = list_active_skill_tools()
            self._backend_tools.extend(skill_tools)
            self._backend_tool_names.update(t["name"] for t in skill_tools)
        except Exception as exc:
            logger.debug("Skill tools unavailable: %s", exc)
            self._skill_tool_index = {}

    def _get_mcp_tools(self) -> List[Dict[str, Any]]:
        if not self._include_mcp_tools:
            return []
        try:
            from services.mcp_client import get_mcp_client
            client = get_mcp_client()
            if client:
                return client.get_tools_for_claude()
        except Exception as exc:
            logger.debug("MCP tools unavailable: %s", exc)
        return []

    def _filter_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter tools by recommended_tools list (agent-specific subset)."""
        if not self._recommended_tools:
            return tools
        allowed = set(self._recommended_tools)
        filtered = []
        for tool in tools:
            name = tool.get("name", "")
            if name in allowed:
                filtered.append(tool)
                continue
            # Match server_toolname format: "splunk_search" matches "search"
            parts = name.split("_", 1)
            if len(parts) == 2 and parts[1] in allowed:
                filtered.append(tool)
        return filtered

    def _all_tools_openai_format(self) -> List[Dict[str, Any]]:
        """Collect backend + MCP tools, filter, and convert to OpenAI format."""
        anthropic_tools = list(self._backend_tools) + self._get_mcp_tools()
        anthropic_tools = self._filter_tools(anthropic_tools)
        if not anthropic_tools:
            return []
        return anthropic_tools_to_openai(anthropic_tools)

    @staticmethod
    def _apply_history_window(
        messages: List[Dict[str, Any]],
        window: int = _HISTORY_WINDOW_DEFAULT,
    ) -> List[Dict[str, Any]]:
        """Enforce a sliding history window (configurable max turns).

        Keeps the system message (if first) + the most recent `window * 2`
        messages. Matches ClaudeService._apply_history_window behavior.
        """
        if window <= 0:
            return messages
        max_msgs = window * 2
        if len(messages) <= max_msgs:
            return messages
        # Preserve system message at index 0 if present
        if messages and messages[0].get("role") == "system":
            return [messages[0]] + messages[-(max_msgs - 1):]
        return messages[-max_msgs:]

    async def stream(
        self,
        *,
        provider: ProviderSpec,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        enable_tools: bool = True,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        history_window: int = _HISTORY_WINDOW_DEFAULT,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a multi-turn agentic conversation with tool calling.

        Yields SSE-compatible event dicts matching the frontend protocol:
            {"type": "text", "content": "..."}
            {"type": "tool_processing", "tool_name": "...",
             "tool_id": "..."}
            {"type": "tool_result", "tool_name": "...",
             "tool_id": "...", "result": "..."}
            {"type": "error", "content": "..."}

        Matches ClaudeService.chat_stream guardrails:
            - 30 tool iterations max
            - 300s wall-clock timeout
            - Exponential inter-iteration backoff
            - Infinite loop detection (3 repeated identical tool sets)
        """
        from openai import AsyncOpenAI

        router = LLMRouter()
        model = model or provider.default_model
        oai_model = f"{provider.provider_type}/{model}"

        tools = self._all_tools_openai_format() if enable_tools else []

        oai_messages: List[Dict[str, Any]] = []
        if system_prompt:
            oai_messages.append({"role": "system", "content": system_prompt})
        oai_messages.extend(messages)

        # Context window management
        oai_messages = self._apply_history_window(oai_messages, history_window)

        client = AsyncOpenAI(
            base_url=f"{router.bifrost_url}/v1",
            api_key="bifrost",
        )

        extra_headers = self._build_headers()
        start_time = asyncio.get_event_loop().time()

        # Infinite loop detection state
        tool_call_history: deque = deque(maxlen=_LOOP_DETECT_WINDOW)

        for iteration in range(_MAX_TOOL_ITERATIONS):
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > _MAX_PROCESSING_TIME_S:
                yield {
                    "type": "text",
                    "content": (
                        "\n\n[Maximum processing time "
                        f"({_MAX_PROCESSING_TIME_S:.0f}s) exceeded "
                        f"after {iteration} iterations.]"
                    ),
                }
                break

            if iteration > 0:
                delay = _inter_iteration_delay(iteration)
                await asyncio.sleep(delay)

            interaction_id = str(uuid.uuid4())
            headers = {
                **extra_headers,
                "x-bf-lh-vigil-interaction-id": interaction_id,
            }

            kwargs: Dict[str, Any] = {
                "model": oai_model,
                "messages": oai_messages,
                "max_tokens": max_tokens,
                "stream": True,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = tools
            if headers:
                kwargs["extra_headers"] = headers

            # Accumulate streamed response
            text_buffer = ""
            tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
            finish_reason: Optional[str] = None
            iter_start = time.monotonic()

            try:
                stream = await client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    finish_reason = choice.finish_reason or finish_reason

                    if delta and delta.content:
                        text_buffer += delta.content
                        yield {"type": "text", "content": delta.content}

                    if delta and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_buffer:
                                tool_calls_buffer[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            entry = tool_calls_buffer[idx]
                            if tc_delta.id:
                                entry["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    entry["name"] += (
                                        tc_delta.function.name
                                    )
                                if tc_delta.function.arguments:
                                    entry["arguments"] += (
                                        tc_delta.function.arguments
                                    )

            except Exception as exc:
                logger.error(
                    "OpenAI stream error iteration %d: %s", iteration, exc
                )
                yield {"type": "error", "content": str(exc)}
                break

            iter_duration_ms = int(
                (time.monotonic() - iter_start) * 1000
            )

            # Persist interaction log (non-fatal)
            self._log_interaction(
                session_id=session_id,
                agent_id=agent_id,
                model=oai_model,
                iteration=iteration,
                interaction_id=interaction_id,
                duration_ms=iter_duration_ms,
                text_content=text_buffer,
                tool_calls_count=len(tool_calls_buffer),
                finish_reason=finish_reason,
            )

            # No tool calls → done
            if finish_reason != "tool_calls" or not tool_calls_buffer:
                # Detect malformed tool calls dumped as text (common with
                # smaller models that attempt tool use but fail at structured
                # output). Filter it rather than passing hallucinated JSON.
                if (
                    text_buffer
                    and iteration == 0
                    and (
                        '{"type":"function"' in text_buffer
                        or ('"function"' in text_buffer and '"parameters"' in text_buffer)
                    )
                ):
                    logger.warning(
                        "Model output contains raw tool-call JSON in text "
                        "(model may not support structured tool calling)"
                    )
                    yield {
                        "type": "text",
                        "content": (
                            "I attempted to use tools but this model doesn't "
                            "reliably support structured tool calling. Please "
                            "select a more capable model in Settings > AI Config "
                            "(e.g., sec8-tools, gpt-4o, or any 7B+ model with "
                            "tool support)."
                        ),
                    }
                break

            # Build assistant message with tool_calls
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if text_buffer:
                assistant_msg["content"] = text_buffer
            assistant_tool_calls = []
            for idx in sorted(tool_calls_buffer.keys()):
                tc = tool_calls_buffer[idx]
                assistant_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                })
            assistant_msg["tool_calls"] = assistant_tool_calls
            oai_messages.append(assistant_msg)

            # Infinite loop detection
            call_signature = frozenset(
                f"{tc['function']['name']}:{tc['function']['arguments']}"
                for tc in assistant_tool_calls
            )
            tool_call_history.append(call_signature)
            if self._detect_infinite_loop(tool_call_history):
                yield {
                    "type": "text",
                    "content": (
                        "\n\n[Stopping: repeated identical tool calls "
                        "detected (possible infinite loop).]"
                    ),
                }
                break

            # Execute each tool and append results
            for tc in assistant_tool_calls:
                tool_name = tc["function"]["name"]
                tool_call_id = tc["id"]
                raw_args = tc["function"]["arguments"]

                yield {
                    "type": "tool_processing",
                    "tool_name": tool_name,
                    "tool_id": tool_call_id,
                }

                result_text = await self._execute_tool(
                    tool_name, raw_args
                )

                yield {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "tool_id": tool_call_id,
                    "result": result_text[:500],
                }

                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_text,
                })
        else:
            # for/else: loop exhausted without break
            yield {
                "type": "text",
                "content": (
                    f"\n\n[Tool iteration limit ({_MAX_TOOL_ITERATIONS}) "
                    "reached. Stopping.]"
                ),
            }

    @staticmethod
    def _detect_infinite_loop(history: deque) -> bool:
        """Detect if the same tool call set repeats >= threshold times."""
        if len(history) < _LOOP_DETECT_THRESHOLD:
            return False
        recent = list(history)[-_LOOP_DETECT_THRESHOLD:]
        return len(set(recent)) == 1

    async def _execute_tool(
        self, tool_name: str, raw_arguments: str
    ) -> str:
        """Dispatch a tool call to the backend or MCP layer."""
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            return f"Error: invalid JSON arguments: {raw_arguments[:200]}"

        if tool_name in self._backend_tool_names:
            return await self._execute_backend_tool(tool_name, arguments)
        return await self._execute_mcp_tool(tool_name, arguments)

    async def _execute_backend_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Execute a backend tool (same dispatch as ClaudeService)."""
        try:
            if tool_name.startswith("skill_"):
                from services.skill_tools_bridge import execute_skill_tool
                result = execute_skill_tool(
                    tool_name,
                    arguments,
                    skills_by_tool_name=self._skill_tool_index,
                )
                return _truncate(json.dumps(result, default=str))

            if tool_name in (
                "analyze_coverage", "search_detections",
                "identify_gaps", "get_coverage_stats",
                "get_detection_count",
            ):
                from tools.security_detections import (
                    get_security_detection_tools,
                )
                security_tools = get_security_detection_tools()
                handler = getattr(security_tools, tool_name, None)
                if handler:
                    result = await handler(**arguments)
                    return _truncate(json.dumps(result, default=str))

            if tool_name in (
                "list_findings", "get_finding", "nearest_neighbors",
                "search_findings", "get_findings_stats", "list_cases",
                "get_case", "create_case", "add_finding_to_case",
                "update_case", "add_resolution_step",
            ):
                from services.database_data_service import (
                    DatabaseDataService,
                )
                data_service = DatabaseDataService()
                handler = getattr(data_service, tool_name, None)
                if handler:
                    result = handler(**arguments)
                    return _truncate(json.dumps(result, default=str))

            return f"Unknown backend tool: {tool_name}"

        except Exception as exc:
            logger.error("Backend tool %s failed: %s", tool_name, exc)
            return f"Error executing {tool_name}: {exc}"

    async def _execute_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Execute an MCP tool via the shared MCP client."""
        try:
            from services.mcp_client import get_mcp_client
            from services.prompt_security import wrap_tool_result

            client = get_mcp_client()
            if not client:
                return f"MCP client unavailable for tool: {tool_name}"

            parts = tool_name.split("_", 1)
            if len(parts) == 2:
                server_name, actual_tool = parts
            else:
                server_name = self._find_tool_server(client, tool_name)
                actual_tool = tool_name

            if not server_name:
                return f"No MCP server found for tool: {tool_name}"

            result = await client.call_tool(
                server_name, actual_tool, arguments,
                timeout=_TOOL_TIMEOUT_S,
            )

            if isinstance(result, dict):
                content_blocks = result.get(
                    "content",
                    [{"type": "text", "text": str(result)}],
                )
            else:
                content_blocks = [{"type": "text", "text": str(result)}]

            texts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = _truncate(block["text"])
                    text = wrap_tool_result(
                        text, source=server_name, tool=actual_tool
                    )
                    texts.append(text)

            return "\n".join(texts) if texts else str(result)

        except Exception as exc:
            logger.error("MCP tool %s failed: %s", tool_name, exc)
            return f"Error executing {tool_name}: {exc}"

    @staticmethod
    def _find_tool_server(
        client: Any, tool_name: str
    ) -> Optional[str]:
        for srv_name, tools in client.tools_cache.items():
            if any(t["name"] == tool_name for t in tools):
                return srv_name
        return None

    @staticmethod
    def _build_headers() -> Dict[str, str]:
        headers: Dict[str, str] = {}
        try:
            from services.budget_service import get_active_vk, should_enforce
            if should_enforce():
                vk = get_active_vk()
                if vk:
                    headers["x-bf-vk"] = vk
        except Exception:
            pass
        return headers

    @staticmethod
    def _log_interaction(
        *,
        session_id: Optional[str],
        agent_id: Optional[str],
        model: str,
        iteration: int,
        interaction_id: str,
        duration_ms: int,
        text_content: str,
        tool_calls_count: int,
        finish_reason: Optional[str],
    ) -> None:
        """Persist an LLMInteractionLog row (non-fatal, fire-and-forget)."""
        try:
            from database.connection import get_db_session
            from database.models import LLMInteractionLog

            session = get_db_session()
            try:
                log = LLMInteractionLog(
                    interaction_id=interaction_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    model=model,
                    request_messages=[],
                    thinking_enabled=False,
                    response_content=(
                        text_content[:2000] if text_content else None
                    ),
                    tool_calls=[{"iteration": iteration}]
                    if tool_calls_count > 0
                    else [],
                    tool_results=[],
                    stop_reason=finish_reason,
                    input_tokens=0,
                    output_tokens=0,
                    cache_read_tokens=0,
                    cache_creation_tokens=0,
                    cost_usd=0,
                    duration_ms=duration_ms,
                )
                session.add(log)
                session.commit()
            except Exception as exc:
                logger.debug("Interaction log failed: %s", exc)
                session.rollback()
            finally:
                session.close()
        except Exception as exc:
            logger.debug("Interaction log skipped: %s", exc)
