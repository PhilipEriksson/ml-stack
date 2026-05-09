import os
import json
import uuid
import httpx

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="ML Stack API")

LLM_SERVER = os.getenv("LLM_SERVER", "http://vllm-server:8000")


async def proxy_get(path):
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.get(f"{LLM_SERVER}{path}")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _stream_chunks(path, payload):
    """Stream from vLLM, yielding text chunks for proper media_type handling."""
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", f"{LLM_SERVER}{path}", json=payload) as resp:
            async for chunk in resp.aiter_text():
                yield chunk


async def proxy_post(path, payload=None, stream=False):
    if stream:
        return StreamingResponse(
            _stream_chunks(path, payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{LLM_SERVER}{path}", json=payload)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


def responses_to_chat(payload):
    """Convert OpenAI Responses API format to chat completions format."""
    messages = []

    instructions = payload.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    inp = payload.get("input", "")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        text_parts = []
        for item in inp:
            if isinstance(item, dict):
                if item.get("type") in ("input_text", "text"):
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and c.get("type") in ("input_text", "text"):
                            text_parts.append(c.get("text", ""))
                    for m in item.get("previous_messages", []):
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") in ("input_text", "text")
                            )
                        messages.append({"role": role, "content": content})
        if text_parts and not any(m["role"] == "user" for m in messages):
            messages.append({"role": "user", "content": "\n".join(text_parts)})

    return {
        "model": payload.get("model"),
        "messages": messages,
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "max_tokens": payload.get("max_output_tokens") or payload.get("max_tokens"),
        "stream": payload.get("stream", False),
    }


def chat_to_responses(result):
    """Convert chat completions response back to Responses API format."""
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    if not result or "choices" not in result:
        return {"id": resp_id, "object": "response", "output": [], "error": result}

    choice = result["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning", "") or ""
    text = f"{reasoning}\n\n{content}".strip() if reasoning else content

    output_item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }

    return {
        "id": resp_id,
        "object": "response",
        "status": "completed",
        "status_details": {},
        "output": [output_item],
        "usage": {
            "input_tokens": result.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": result.get("usage", {}).get("completion_tokens", 0),
        },
    }


def stream_chat_to_responses_events(result_lines):
    """Convert streaming chat completion SSE events to Responses API SSE events."""
    for line in result_lines:
        if not line.strip():
            yield f"{line}\n"
            continue
        if line.startswith("data: ") and line.strip() == "data: [DONE]":
            yield f"{line}\n"
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            try:
                data = json.loads(data_str)
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        resp_data = {
                            "type": "response.output_text.delta",
                            "output_text": {"annotations": [], "delta": {"type": "output_text.delta", "value": content}},
                        }
                        yield f"data: {json.dumps(resp_data)}\n\n"
            except json.JSONDecodeError:
                yield f"{line}\n"
            continue
        yield f"{line}\n"


@app.get("/")
async def root():
    return {"status": "ok", "service": "ml-stack-api"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models_v1():
    return await proxy_get("/v1/models")


@app.get("/models")
async def models():
    return await proxy_get("/v1/models")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    return await proxy_post("/v1/chat/completions", payload, stream=payload.get("stream", False))


@app.post("/v1/completions")
async def completions(request: Request):
    payload = await request.json()
    return await proxy_post("/v1/completions", payload, stream=payload.get("stream", False))


@app.post("/v1/responses")
async def responses(request: Request):
    payload = await request.json()
    chat_payload = responses_to_chat(payload)
    chat_payload.pop("stream", None)

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{LLM_SERVER}/v1/chat/completions", json=chat_payload)
        result = resp.json()
        return JSONResponse(content=chat_to_responses(result), status_code=resp.status_code)


# Catch-all for any other /v1/* routes
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_v1(path: str, request: Request):
    if request.method == "GET":
        return await proxy_get(f"/v1/{path}")
    else:
        payload = await request.json() if request.body else None
        return await proxy_post(f"/v1/{path}", payload)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_catchall(path: str, request: Request):
    if request.method == "GET":
        return await proxy_get(f"/{path}")
    else:
        payload = await request.json() if request.body else None
        return await proxy_post(f"/{path}", payload)
