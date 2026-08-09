"""
Mock OpenAI-compatible API server.
Run:    python3 server.py
Query:  curl http://localhost:8000/v1/chat/completions ...
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time, uuid, json, random

app = FastAPI(title="Mock OpenAI-Compatible API")

# ---- Mock data ----
MOCK_REPLIES = [
    "Hello! This is a mock response from your local API.",
    "I'm just canned text right now, but the shape of this response matches OpenAI's API.",
    "This response is fake, but the JSON structure is real and swappable with your own model later.",
]

MOCK_MODELS = ["mock-gpt-4o", "mock-gpt-3.5-turbo"]


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = 1.0


def mock_reply(messages: list[Message]) -> str:
    # Trivial "logic": echo the last user message inside a canned reply
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    return f"{random.choice(MOCK_REPLIES)} (you said: '{last_user}')"


@app.get("/v1/openai/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock-org",
            }
            for m in MOCK_MODELS
        ],
    }


@app.post("/v1/openai/chat/completions")
async def chat_completions(req: ChatRequest):
    reply_text = mock_reply(req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:

        def event_stream():
            # send role first, like real OpenAI streaming
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            for word in reply_text.split(" "):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                time.sleep(0.05)

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.content.split()) for m in req.messages),
            "completion_tokens": len(reply_text.split()),
            "total_tokens": 0,  # left as mock; real servers sum the two above
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
