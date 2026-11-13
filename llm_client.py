"""Provider-independent LLM communication using standard-library HTTP."""
import json
import urllib.request
import urllib.error


class LLMRequestError(RuntimeError):
    def __init__(self, message, *, retryable):
        super().__init__(message)
        self.retryable = retryable


class LLMHTTPError(LLMRequestError):
    def __init__(self, status, body):
        super().__init__(f'LLM API HTTP {status}: {body}',
                         retryable=status in (408, 409, 429) or status >= 500)
        self.status = status
        self.body = body


class LLMClient:
    def __init__(self, base_url, api_key, model, timeout=600, wire_api="chat", timeout_option="--llm-timeout"):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        self.wire_api = wire_api   # chat=OpenAI /chat/completions; responses=GPT /responses
        self.timeout_option = timeout_option

    def chat(self, messages, temperature=1, max_tokens=None, reasoning_effort=None):
        """Run a single- or multi-turn conversation and return assistant text. Each modeling file has an independent conversation."""
        if self.wire_api == "responses":
            payload = {
                "model": self.model,
                "stream": False,
                "input": [{"role": m["role"], "content": m["content"]}
                          for m in messages],
                "temperature": temperature,
            }
            if reasoning_effort:
                payload["reasoning"] = {"effort": reasoning_effort}
            if max_tokens:
                payload["max_output_tokens"] = max_tokens
            data = self._post(f"{self.base_url}/responses", payload)
            if data.get("output_text"):
                return data["output_text"]
            parts = []
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text"):
                            parts.append(c.get("text", ""))
            if not parts:
                raise LLMRequestError(f"No text output found in Responses reply: {str(data)[:300]}", retryable=True)
            return "".join(parts)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if max_tokens:
            payload["max_tokens"] = max_tokens
        data = self._post(f"{self.base_url}/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError('assistant content must be text')
            return text
        except (KeyError, IndexError, TypeError) as error:
            raise LLMRequestError('Invalid LLM reply: ' + str(error), retryable=True) from error

    def _post(self, url, payload):
        try:
            return self._post_once(url, payload)
        except LLMHTTPError as e:
            # Reasoning models (such as the GPT-5 family) may reject temperature with HTTP 400.
            # If the error mentions temperature, remove that field and retry once.
            if e.status == 400 and "temperature" in e.body and "temperature" in payload:
                payload = {k: v for k, v in payload.items() if k != "temperature"}
                return self._post_once(url, payload)
            raise

    def _post_once(self, url, payload):
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            raise LLMHTTPError(e.code, body) from e
        except json.JSONDecodeError as e:
            raise LLMRequestError('LLM API returned invalid JSON: ' + str(e), retryable=True) from e
        except (urllib.error.URLError, TimeoutError) as e:
            reason = str(e)
            if "timed out" in reason or "timeout" in reason.lower():
                raise LLMRequestError(
                    f"LLM request timed out (>{self.timeout}s; adjust with {self.timeout_option})", retryable=True) from e
            raise LLMRequestError(f"LLM network error (not a timeout): {reason}", retryable=True) from e
