"""
Chat UI component for the Trading Dashboard.

Renders a chat interface with Claude tool-use loop (Anthropic API).
Uses Streamlit chat_message / chat_input with session state history.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.data.chat_tools import define_chat_tools, execute_tool
from dashboard.data.chat_context import build_system_prompt

log = logging.getLogger(__name__)


def _get_anthropic_client():
    """Get or create a cached Anthropic client."""
    api_key = _resolve_api_key()
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    except ImportError:
        return None
    except Exception as e:
        log.error(f"Error creating Anthropic client: {e}")
        return None


def get_chat_response(
    question: str,
    chat_history: list,
    system_prompt: str,
    tools: list,
) -> Tuple[str, list]:
    """
    Claude tool-use loop (max 10 iterations).

    Returns:
        Tuple of (response_text, tool_results_display_list)
    """
    client = _get_anthropic_client()
    if client is None:
        return "Error: Anthropic API not configured. Set ANTHROPIC_API_KEY environment variable.", []

    # Build messages from recent history + current question
    messages = []
    for msg in chat_history[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    max_iterations = 10
    iteration = 0
    tool_results_display = []

    while iteration < max_iterations:
        iteration += 1

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                system=system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=4096,
            )
        except Exception as e:
            log.error(f"Anthropic API error: {e}")
            return f"Error calling Claude API: {str(e)}", tool_results_display

        if response.stop_reason == "tool_use":
            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    result = execute_tool(tool_name, tool_input)

                    tool_results_display.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": result,
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Extract text response
            text_parts = [b.text for b in response.content if hasattr(b, 'text')]
            return "\n".join(text_parts), tool_results_display

    return "Maximum iterations reached. The assistant made too many tool calls.", tool_results_display


def _resolve_api_key() -> str | None:
    """Get Anthropic API key from env or session state."""
    return os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("chat_api_key")


def render_chat(report_data=None):
    """Render the chat interface."""

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build tools list (always available for display)
    tools = define_chat_tools()

    # Tool availability info
    from dashboard.data.chat_tools import OBSIDIAN_AVAILABLE, MONGO_AVAILABLE, SEMANTIC_AVAILABLE
    available_tools = len(tools)
    integrations = []
    if OBSIDIAN_AVAILABLE:
        integrations.append("Obsidian")
    if MONGO_AVAILABLE:
        integrations.append("MongoDB")
    if SEMANTIC_AVAILABLE:
        integrations.append("Semantic")

    with st.sidebar:
        st.markdown("#### Chat Settings")

        # API key input if not in environment
        if not os.getenv("ANTHROPIC_API_KEY"):
            api_key_input = st.text_input(
                "Anthropic API Key",
                type="password",
                key="chat_api_key_input",
                placeholder="sk-ant-...",
            )
            if api_key_input:
                st.session_state.chat_api_key = api_key_input

        if integrations:
            st.caption(f"Integrations: {', '.join(integrations)}")
        else:
            st.caption("Backtester tools only (no MCP)")
        st.caption(f"{available_tools} tools available")

        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                with st.expander(f"Tools used ({len(msg['tools_used'])})"):
                    for t in msg["tools_used"]:
                        st.caption(f"**{t['tool']}**")
                        # Format input params
                        input_str = ", ".join(f"{k}={v}" for k, v in t['input'].items())
                        if input_str:
                            st.caption(f"  Input: {input_str}")
                        # Show truncated result
                        try:
                            parsed = json.loads(t['result'])
                            st.code(json.dumps(parsed, indent=2)[:800], language="json")
                        except (json.JSONDecodeError, TypeError):
                            st.code(str(t['result'])[:800])

    # Build system prompt (deferred until needed)
    system_prompt = build_system_prompt(report_data)

    # Chat input — always visible
    if prompt := st.chat_input("Ask about your trades..."):
        # Check for API key at send time
        if not _resolve_api_key():
            st.error("Enter your Anthropic API key in the sidebar to use chat.")
            st.stop()

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, tools_used = get_chat_response(
                    prompt,
                    st.session_state.chat_history[:-1],
                    system_prompt,
                    tools,
                )

            st.markdown(response)

            if tools_used:
                with st.expander(f"Tools used ({len(tools_used)})"):
                    for t in tools_used:
                        st.caption(f"**{t['tool']}**")
                        input_str = ", ".join(f"{k}={v}" for k, v in t['input'].items())
                        if input_str:
                            st.caption(f"  Input: {input_str}")
                        try:
                            parsed = json.loads(t['result'])
                            st.code(json.dumps(parsed, indent=2)[:800], language="json")
                        except (json.JSONDecodeError, TypeError):
                            st.code(str(t['result'])[:800])

        # Save assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "tools_used": tools_used,
        })
