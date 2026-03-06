"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """OpenRouter API wrapper. All LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None
        self._gigachat_token = None
        self._gigachat_token_expires = 0.0
        self._last_gigachat_token = None

    def _update_gigachat_token(self):
        import requests
        import uuid
        import urllib3
        
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        auth_data = os.environ.get("GIGACHAT_CREDENTIALS")
        if not auth_data:
            raise ValueError("GIGACHAT_CREDENTIALS environment variable is required for GigaChat models")
            
        scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': str(uuid.uuid4()),
            'Authorization': f'Basic {auth_data}'
        }
        
        resp = requests.post(
            'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
            headers=headers,
            data={'scope': scope},
            verify=False,
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        self._gigachat_token = data.get('access_token')
        
        # expires_at is typically unix epoch in milliseconds
        expires_at = data.get('expires_at', time.time() * 1000 + 1800000)
        self._gigachat_token_expires = expires_at / 1000.0

    def _get_client(self, model: str = ""):
        if model.startswith("gigachat/"):
            import time
            now = time.time()
            if not self._gigachat_token or now >= self._gigachat_token_expires - 60:
                self._update_gigachat_token()
                
            if not getattr(self, "_gigachat_openai_client", None) or self._last_gigachat_token != self._gigachat_token:
                from openai import OpenAI
                import httpx
                http_client = httpx.Client(verify=False)
                self._gigachat_openai_client = OpenAI(
                    base_url="https://gigachat.devices.sberbank.ru/api/v1",
                    api_key=self._gigachat_token,
                    http_client=http_client,
                )
                self._last_gigachat_token = self._gigachat_token
            return self._gigachat_openai_client

        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://colab.research.google.com/",
                    "X-Title": "Ouroboros",
                },
            )
        return self._client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        client = self._get_client(model)
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {
            "reasoning": {"effort": effort, "exclude": True},
        }

        # Pin Anthropic models to Anthropic provider for prompt caching
        if model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        
        actual_model = model
        if model.startswith("gigachat/"):
            actual_model = model[len("gigachat/"):]
            extra_body.clear() # GigaChat does not support OpenRouter extra_body
            
            # GigaChat does not support: list-typed content, role="tool",
            # or assistant messages with tool_calls. Flatten all of these.
            def _flatten_content_for_gigachat(content):
                """Convert list content to plain string for GigaChat."""
                if isinstance(content, str):
                    return content
                if not isinstance(content, list):
                    return str(content) if content else ""
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(str(block.get("text") or ""))
                        elif block.get("type") == "tool_result":
                            inner = block.get("content") or ""
                            if isinstance(inner, list):
                                for ib in inner:
                                    if isinstance(ib, dict) and ib.get("type") == "text":
                                        parts.append(str(ib.get("text") or ""))
                            elif isinstance(inner, str):
                                parts.append(inner)
                return " ".join(parts) if parts else ""

            new_messages = []
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")

                if role == "tool":
                    # GigaChat doesn't understand role=tool; convert to user message
                    tool_name = m.get("name") or m.get("tool_call_id") or "tool"
                    text = _flatten_content_for_gigachat(content)
                    new_messages.append({
                        "role": "user",
                        "content": f"[Tool result from {tool_name}]: {text}",
                    })
                elif role == "assistant" and m.get("tool_calls"):
                    # Strip tool_calls; summarize as plain text if no content
                    text = _flatten_content_for_gigachat(content)
                    tool_summaries = []
                    for tc in (m.get("tool_calls") or []):
                        fn = (tc.get("function") or {}).get("name", "tool")
                        args = (tc.get("function") or {}).get("arguments", "")
                        tool_summaries.append(f"[Called {fn}({args[:100]})]")
                    combined = " ".join(filter(None, [text] + tool_summaries))
                    new_messages.append({"role": "assistant", "content": combined or "[tool call]"})
                else:
                    new_m = {**m}
                    new_m["content"] = _flatten_content_for_gigachat(content)
                    new_messages.append(new_m)
            messages = new_messages

        kwargs: Dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        if tools:
            # Add cache_control to last tool for Anthropic prompt caching
            # This caches all tool schemas (they never change between calls)
            tools_with_cache = [t for t in tools]  # shallow copy
            if not model.startswith("gigachat/") and tools_with_cache:
                last_tool = {**tools_with_cache[-1]}  # copy last tool
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage (OpenRouter includes it, but fallback if missing)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
