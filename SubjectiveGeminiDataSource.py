import os
import mimetypes
import time
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional
import requests
from subjective_abstract_data_source_package import SubjectiveOnDemandDataSource
from brainboost_data_source_logger_package.BBLogger import BBLogger


class SubjectiveGeminiDataSource(SubjectiveOnDemandDataSource):
    def __init__(self, name=None, session=None, dependency_data_sources=None,
                 subscribers=None, params=None):
        super().__init__(
            name=name,
            session=session,
            dependency_data_sources=dependency_data_sources,
            subscribers=subscribers,
            params=params
        )
        self.api_key = self.params.get("api_key")
        self.model = self.params.get("model", "gemini-2.0-flash")
        self.temperature = self.params.get("temperature")
        self.max_tokens = self.params.get("max_tokens")
        self.system_prompt = (self.params.get("system_prompt") or "").strip()
        self.api_base_url = (self.params.get("api_base_url") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.timeout = self.params.get("timeout")
        self.rate_limit_per_minute = self._coerce_positive_number(self.params.get("rate_limit_per_minute"), default=60)
        self.max_retries = int(self._coerce_positive_number(self.params.get("max_retries"), default=3))
        self.retry_backoff_base = self._coerce_positive_number(self.params.get("retry_backoff_base"), default=1.0)
        self.retry_backoff_max = self._coerce_positive_number(self.params.get("retry_backoff_max"), default=30.0)
        self._rate_lock = threading.Lock()
        self._last_request_ts = 0.0

    def _process_message(self, message: Any) -> Any:
        """
        Process an incoming message and return a response from Google Gemini.

        Args:
            message: The incoming message to process

        Returns:
            The response from Gemini
        """
        BBLogger.log(f"Processing message: {str(message)[:100]}...")

        if isinstance(message, dict) and message.get("files"):
            return self._process_message_with_files(message)

        if not self.api_key:
            return {"error": "Missing Gemini API key", "message": message}

        # Build contents array with conversation history
        contents: List[Dict[str, Any]] = []

        # Include recent conversation history
        for entry in self._conversation_history[-20:]:
            role = entry.get("role")
            if role not in ("user", "assistant"):
                continue
            content = entry.get("content")
            # Map assistant to model for Gemini API
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": str(content)}]
            })

        payload: Dict[str, Any] = {
            "contents": contents
        }

        # Add generation config
        generation_config: Dict[str, Any] = {}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.max_tokens:
            generation_config["maxOutputTokens"] = self.max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        # Add system instruction if provided
        if self.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": self.system_prompt}]
            }

        url = f"{self.api_base_url}/models/{self.model}:generateContent?key={self.api_key}"
        timeout = 60
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:
            timeout = self.timeout

        headers = {
            "Content-Type": "application/json"
        }

        data = self._send_gemini_request(url, headers, payload, timeout)

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return data

    def _process_message_with_files(self, message: Dict[str, Any]) -> Any:
        if not self.api_key:
            return {"error": "Missing Gemini API key", "message": message}

        user_text = str(message.get("content") or "")
        files = self._normalize_files(message.get("files"))

        # Build contents array with conversation history
        contents: List[Dict[str, Any]] = []

        for entry in self._conversation_history[-20:]:
            role = entry.get("role")
            if role not in ("user", "assistant"):
                continue
            content = entry.get("content")
            if isinstance(content, dict):
                content = content.get("content") or content.get("message") or ""
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": str(content)}]
            })

        # Build user message with files
        user_parts = self._build_gemini_parts(user_text, files)
        contents.append({
            "role": "user",
            "parts": user_parts
        })

        payload: Dict[str, Any] = {
            "contents": contents
        }

        # Add generation config
        generation_config: Dict[str, Any] = {}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.max_tokens:
            generation_config["maxOutputTokens"] = self.max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        # Add system instruction if provided
        if self.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": self.system_prompt}]
            }

        url = f"{self.api_base_url}/models/{self.model}:generateContent?key={self.api_key}"
        timeout = 60
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:
            timeout = self.timeout

        headers = {
            "Content-Type": "application/json"
        }

        data = self._send_gemini_request(url, headers, payload, timeout)

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return data

    def _send_gemini_request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Any:
        total_attempts = max(1, self.max_retries + 1)
        for attempt in range(total_attempts):
            self._enforce_rate_limit()
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.exceptions.RequestException as e:
                if attempt >= total_attempts - 1:
                    return {
                        "error": "Gemini request failed",
                        "details": str(e)
                    }
                backoff = self._compute_backoff(attempt)
                BBLogger.log(f"[SubjectiveGeminiDataSource] Request error; retrying in {backoff:.1f}s: {e}")
                time.sleep(backoff)
                continue

            if response.status_code == 429:
                retry_after = self._get_retry_after_seconds(response)
                wait_seconds = retry_after if retry_after is not None else self._compute_backoff(attempt)
                BBLogger.log(
                    "[SubjectiveGeminiDataSource] Rate limit hit (429). "
                    f"Retrying in {wait_seconds:.1f}s (attempt {attempt + 1}/{total_attempts})."
                )
                if attempt >= total_attempts - 1:
                    return {
                        "error": "Gemini rate limit exceeded",
                        "details": "Please wait before sending more requests.",
                        "retry_after_seconds": wait_seconds
                    }
                time.sleep(wait_seconds)
                continue

            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if attempt >= total_attempts - 1:
                    return {
                        "error": "Gemini API error",
                        "details": str(e),
                        "status_code": response.status_code
                    }
                backoff = self._compute_backoff(attempt)
                BBLogger.log(
                    "[SubjectiveGeminiDataSource] HTTP error; "
                    f"retrying in {backoff:.1f}s: {e}"
                )
                time.sleep(backoff)
                continue

            return response.json()

        return {
            "error": "Gemini request failed",
            "details": "Maximum retries exceeded"
        }

    def _enforce_rate_limit(self):
        if not self.rate_limit_per_minute or self.rate_limit_per_minute <= 0:
            return
        min_interval = 60.0 / float(self.rate_limit_per_minute)
        with self._rate_lock:
            now = time.time()
            wait_seconds = min_interval - (now - self._last_request_ts)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.time()
            self._last_request_ts = now

    def _compute_backoff(self, attempt: int) -> float:
        base = float(self.retry_backoff_base or 1.0)
        max_backoff = float(self.retry_backoff_max or 30.0)
        return min(max_backoff, base * (2 ** attempt))

    def _get_retry_after_seconds(self, response: requests.Response) -> Optional[float]:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                dt = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())

    def _coerce_positive_number(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if number <= 0:
            return default
        return number

    def _normalize_files(self, files: Any) -> List[Dict[str, Any]]:
        if not isinstance(files, list):
            return []
        normalized = []
        for item in files:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

    def _build_gemini_parts(self, user_text: str, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build Gemini parts array from text and files."""
        parts: List[Dict[str, Any]] = []

        for payload in files:
            mime_type = payload.get("mime_type") or self._guess_mime_type(payload.get("name"))
            data_base64 = payload.get("data_base64")

            # Handle images and other inline data
            if mime_type and data_base64:
                if mime_type.startswith("image/"):
                    parts.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": data_base64
                        }
                    })
                    continue

            # Handle text content from files
            text_block = self._format_file_text(payload)
            if text_block:
                parts.append({"text": text_block})

        # Add user text at the end
        if user_text:
            parts.append({"text": user_text})

        if not parts:
            parts.append({"text": ""})

        return parts

    def _format_file_text(self, payload: Dict[str, Any]) -> str:
        name = payload.get("name") or "attachment"
        mime_type = payload.get("mime_type") or self._guess_mime_type(name) or "application/octet-stream"
        size = payload.get("size")
        text = payload.get("text")
        if isinstance(text, str) and text:
            return self._truncate_text(
                f"[Attached file: {name} | {mime_type} | {size} bytes]\n{text}"
            )

        data_base64 = payload.get("data_base64")
        if isinstance(data_base64, str) and data_base64:
            snippet = self._truncate_text(data_base64, max_chars=10000)
            return f"[Attached file: {name} | {mime_type} | {size} bytes | base64]\n{snippet}"

        return f"[Attached file: {name} | {mime_type} | {size} bytes | no content provided]"

    def _truncate_text(self, text: str, max_chars: int = 20000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[truncated]"

    def _guess_mime_type(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        mime_type, _ = mimetypes.guess_type(name)
        return mime_type

    def get_icon(self) -> str:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        try:
            with open(icon_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            BBLogger.log(f"Error reading icon file: {e}")
            return ""

    def get_connection_data(self) -> dict:
        """Return connection configuration metadata for Google Gemini API."""
        return {
            "connection_type": "ON_DEMAND",
            "fields": [
                {
                    "name": "connection_name",
                    "type": "text",
                    "label": "Connection Name",
                    "required": True,
                    "description": "A friendly name for this Gemini connection"
                },
                {
                    "name": "api_key",
                    "type": "password",
                    "label": "Gemini API Key",
                    "required": True,
                    "description": "Your Google Gemini API key from Google AI Studio"
                },
                {
                    "name": "model",
                    "type": "select",
                    "label": "Model",
                    "required": True,
                    "default": "gemini-2.0-flash",
                    "options": [
                        {"value": "gemini-2.0-flash", "label": "Gemini 2.0 Flash (Recommended)"},
                        {"value": "gemini-2.0-flash-lite", "label": "Gemini 2.0 Flash Lite (Faster)"},
                        {"value": "gemini-1.5-pro", "label": "Gemini 1.5 Pro"},
                        {"value": "gemini-1.5-flash", "label": "Gemini 1.5 Flash"},
                        {"value": "gemini-1.5-flash-8b", "label": "Gemini 1.5 Flash 8B (Cheapest)"}
                    ]
                },
                {
                    "name": "temperature",
                    "type": "number",
                    "label": "Temperature",
                    "required": False,
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "Controls randomness: 0 = deterministic, 2 = very random"
                },
                {
                    "name": "max_tokens",
                    "type": "number",
                    "label": "Max Output Tokens",
                    "required": False,
                    "default": 8192,
                    "min": 1,
                    "max": 8192,
                    "description": "Maximum number of tokens in the response"
                },
                {
                    "name": "system_prompt",
                    "type": "textarea",
                    "label": "System Prompt",
                    "required": False,
                    "default": "You are a helpful assistant.",
                    "description": "Instructions that define the assistant's behavior"
                },
                {
                    "name": "api_base_url",
                    "type": "text",
                    "label": "API Base URL",
                    "required": False,
                    "default": "https://generativelanguage.googleapis.com/v1beta",
                    "description": "Custom API endpoint (for proxies or different regions)"
                },
                {
                    "name": "timeout",
                    "type": "number",
                    "label": "Timeout (seconds)",
                    "required": False,
                    "default": 60,
                    "min": 10,
                    "max": 300,
                    "description": "Maximum time to wait for API response"
                },
                {
                    "name": "rate_limit_per_minute",
                    "type": "number",
                    "label": "Requests Per Minute",
                    "required": False,
                    "default": 60,
                    "min": 1,
                    "max": 600,
                    "description": "Client-side rate limit for Gemini requests"
                },
                {
                    "name": "max_retries",
                    "type": "number",
                    "label": "Max Retries",
                    "required": False,
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "description": "Maximum retry attempts for transient errors"
                },
                {
                    "name": "retry_backoff_base",
                    "type": "number",
                    "label": "Retry Backoff Base (seconds)",
                    "required": False,
                    "default": 1.0,
                    "min": 0.5,
                    "max": 10,
                    "description": "Base delay for exponential backoff"
                },
                {
                    "name": "retry_backoff_max",
                    "type": "number",
                    "label": "Retry Backoff Max (seconds)",
                    "required": False,
                    "default": 30.0,
                    "min": 1,
                    "max": 120,
                    "description": "Maximum delay for exponential backoff"
                }
            ]
        }
