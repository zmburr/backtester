"""
LLMClient v2 – centrally routes 4 logical "tiers" of models to the
right provider with graceful fallback.

Tiers
-----
1. fast_foundation  – very low‑latency, moderate reasoning
2. fast_thinking    – low‑latency but bigger context / better reasoning
3. smart_foundation – slower but strong reasoning
4. smart_thinking   – highest quality

Default routing
---------------
fast_foundation  -> Cerebras (llama‑3‑8B‑instruct)  → fallback Groq (gemma‑7b)
fast_thinking    -> Groq (llama‑70b‑r1)            → fallback Cerebras
smart_foundation -> Together (Meta‑Llama‑3.1‑70B‑Turbo) → fallback OpenAI gpt‑4o‑mini
smart_thinking   -> OpenAI gpt‑4o                   → fallback Together (Mistral‑8x22B)

Override the default model or provider via kwargs.
"""
from __future__ import annotations
import os
from support.config import OPENAI_API_KEY, TOGETHER_API_KEY, CEREBRAS_API_KEY, GROQ_API_KEY, PPLX_API_KEY
from groq import Groq
from typing import Union, List, Dict
import asyncio
import logging
from typing import Sequence, Dict, List, Literal, Callable, Any
import inspect
try:
    # modern OpenAI SDK (>=1.0)
    from openai import OpenAI
except Exception:             # pragma: no cover - optional dependency
    OpenAI = None
    openai_legacy = None
else:
    # fallback for legacy openai<1.0 which exposes module level API
    try:
        import openai as openai_legacy
    except Exception:         # pragma: no cover - optional dependency
        openai_legacy = None
from together import Together
from cerebras.cloud.sdk import Cerebras
import re as _re
import requests  # for Perplexity API client


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s – %(message)s")

def _strip_think(text: str) -> str:
    """Remove the <think>…</think> block if present."""
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.S).strip()


# -------------------- model routing table --------------------
Tier = Literal["fast_foundation", "fast_thinking",
               "fast_foundation_fallback", "smart_foundation", "smart_thinking"]

_DEFAULTS: Dict[Tier, List[tuple[str, str]]] = {
    "fast_foundation": [
        ("cerebras", os.getenv("FAST_FOUND_CEREBRAS_MODEL", "llama3.3-70b")),
    ],
    "fast_foundation_fallback": [
        ("openai", os.getenv("FAST_FOUND_FALLBACK_OAI_MODEL", "gpt-4.1-mini")),
    ],
    "fast_thinking": [
        ("groq", os.getenv("FAST_THINK_GROQ_MODEL", "deepseek-r1-distill-llama-70b")),
    ],
    "smart_foundation": [
        ("openai",   os.getenv("SMART_FOUND_OAI_MODEL", "gpt-4.1")),
    ],
    "smart_thinking": [
        ("openai", os.getenv("SMART_THINK_OAI_MODEL", "o3")),
    ],
}

# ───────────────────────── provider factory ───────────────────────────────
def _build_client(provider: str) -> Callable[..., Any]:
    if provider == "openai":
        if OpenAI:                              # modern SDK
            return OpenAI(api_key=OPENAI_API_KEY).chat.completions.create
        if openai_legacy:                       # legacy SDK
            openai_legacy.api_key = OPENAI_API_KEY
            return openai_legacy.ChatCompletion.create  # type: ignore[attr-defined]
        raise RuntimeError("openai SDK not installed")

    if provider == "together":
        if not Together:
            raise RuntimeError("together SDK missing")
        return Together(api_key=TOGETHER_API_KEY).chat.completions.create

    if provider == "cerebras":
        if not Cerebras:
            raise RuntimeError("cerebras SDK missing")
        return Cerebras(api_key=CEREBRAS_API_KEY).chat.completions.create

    if provider == "groq":
        client = Groq(api_key=GROQ_API_KEY)
        return client.chat.completions.create

    raise ValueError(f"unknown provider '{provider}'")


_clients: Dict[str, Callable[..., Any]] = {}
for routes in _DEFAULTS.values():
    for provider, _ in routes:
        if provider not in _clients:
            try:
                _clients[provider] = _build_client(provider)
            except Exception as exc:
                log.warning("Provider %s unavailable: %s", provider, exc)

# ─────────────────────────── helpers ──────────────────────────────────────
import re as _re
def _strip_think(text: str) -> str:
    """Remove <think>…</think> block (Groq DeepSeek, etc.)."""
    return _re.sub(r"<think>.*?</think>", "", text, flags=_re.S).strip()

# ─────────────────────────── main façade ─────────────────────────────────
class LLMClientV2:
    async def chat(
        self,
        messages: Sequence[dict],
        tier: Tier = "smart_foundation",
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str | None:
        """Route to first available provider in the tier."""
        for provider, default_model in _DEFAULTS[tier]:
            if provider not in _clients:
                continue

            chosen_model = model or default_model
            send_messages = [m.copy() for m in messages]

            # o-series guard-rail: remap system → developer
            if provider == "openai" and chosen_model.lower().startswith("o"):
                for msg in send_messages:
                    if msg.get("role") == "system":
                        msg["role"] = "developer"

            try:
                txt = await self._call_async(
                    _clients[provider],
                    provider=provider,
                    model=chosen_model,
                    messages=send_messages,
                    temperature=temperature,
                )
                log.info("✓ %s:%s served tier %s", provider, chosen_model, tier)
                return txt
            except Exception as exc:
                log.warning("✗ %s failed for tier %s – %s", provider, tier, exc)
        # If every provider in the current tier failed, optionally fall back to a safer tier.
        # For now we only handle the case where fast_foundation exhausts its providers and we want
        # to seamlessly retry using the designated fast_foundation_fallback tier.
        if tier == "fast_foundation":
            log.info("All providers for 'fast_foundation' failed; falling back to 'fast_foundation_fallback'.")
            return await self.chat(
                messages=messages,
                tier="fast_foundation_fallback",
                temperature=temperature,
                model=model,
            )

        return None

    # ------------------------------------------------------------------
    @staticmethod
    async def _call_async(
        fn: Callable[..., Any],
        provider: str,
        model: str,
        **kwargs,
    ) -> str:
        """Run blocking SDK call in executor; normalise response."""
        # o-series rejects explicit temperature other than default (1)
        if provider == "openai" and model.lower().startswith("o"):
            kwargs.pop("temperature", None)

        loop = asyncio.get_running_loop()

        def _sync():
            resp = fn(model=model, **kwargs)
            content = (
                resp.choices[0].message.content
                if hasattr(resp, "choices")
                else resp["choices"][0]["message"]["content"]
            )
            return _strip_think(content)

        return await loop.run_in_executor(None, _sync)


llm = LLMClientV2()


def openai_search_response(
    messages: Union[str, List[Dict]],
    temperature: float = 1,
    max_tokens: int = 2048,
    location: dict | None = None,
    store: bool = False
):
    # if they passed a string, wrap it as a single user message
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    input_blocks = []
    for m in messages:
        block = {
            "role": m["role"],
            "content": [{
                "type": "input_text",
                "text": m["content"].strip()
            }]
        }
        input_blocks.append(block)

    if location is None:
        location = {
            "type": "approximate",
            "country": "US", "region": "NY", "city": "New York"
        }

    response = OpenAI(api_key=OPENAI_API_KEY).responses.create(
        model="gpt-4.1-mini",
        input=input_blocks,
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[{
            "type": "web_search_preview",
            "user_location": location,
            "search_context_size": "medium"
        }],
        temperature=temperature,
        max_output_tokens=max_tokens,
        top_p=1,
        store=store
    )
    return response.output_text.strip()


def perplexity_search(query: str, model: str = "sonar", system_prompt: str = "You are an expert in news and current events. You are given a headline and you need to search the web for the most recent and relevant information related to the headline.") -> dict:
    """
    Search Perplexity API for new and relevant information.
    """
    if not PPLX_API_KEY:
        raise RuntimeError("PPLX_API_KEY environment variable not set")
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'What is the latest news related to the headline: {query}'}
        ]
    }
    headers = {
        "Authorization": f"Bearer {PPLX_API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.request("POST", url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


# ────────────────────── smoke-test harness ───────────────────────────────
if __name__ == "__main__":

    async def smoke():
        test = [
            {"role": "system", "content": "You are a calculator."},
            {"role": "user",   "content": "2 + 2 = ?"},
        ]
        response = await llm.chat(test, tier="smart_foundation")

        print(response)
    asyncio.run(smoke())

def chat(messages, tier="smart_foundation", **kw):
    caller = inspect.stack()[1]
    loc    = f"{caller.filename}:{caller.lineno} in {caller.function}"
    logging.info("🔹 llm.chat invoked from %s | tier=%s", loc, tier)
    return _real_chat(messages, tier=tier, **kw)   # existing implementation