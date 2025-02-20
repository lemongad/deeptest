from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator, List, Optional
import secrets
import time
import uuid
import hashlib
import json
import re
import httpx

app = FastAPI()

# ========== 数据模型 ==========
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "deepseek"
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

# ========== 工具类 ==========
class SessionTools:
    @staticmethod
    def gen_device_id() -> str:
        return f"{uuid.uuid4().hex}_{SessionTools.nanoid(20)}"

    @staticmethod
    def nanoid(size=21) -> str:
        alphabet = 'useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict'
        return ''.join(alphabet[secrets.randbelow(64)] for _ in range(size))

class AuthGenerator:
    @staticmethod
    def create_sign(timestamp: str, payload: dict, nonce: str) -> str:
        sorted_payload = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
        sign_str = f"{timestamp}{sorted_payload}{nonce}"
        return hashlib.md5(sign_str.encode()).hexdigest().upper()

# ========== 核心服务 ==========
class DeepSeekService:
    API_BASE = "https://ai-api.dangbei.net"
    DEFAULT_HEADERS = {
        "Origin": "https://ai.dangbei.com",
        "Referer": "https://ai.dangbei.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    def __init__(self):
        self.session_tools = SessionTools()
        self.auth_tools = AuthGenerator()

    async def _init_conversation(self, device_id: str) -> str:
        """初始化新会话"""
        timestamp = str(int(time.time()))
        nonce = SessionTools.nanoid(21)
        payload = {"botCode": "AI_SEARCH"}
        
        headers = {
            **self.DEFAULT_HEADERS,
            "deviceId": device_id,
            "nonce": nonce,
            "sign": self.auth_tools.create_sign(timestamp, payload, nonce),
            "timestamp": timestamp
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.API_BASE}/ai-search/conversationApi/v1/create",
                    json=payload,
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()["data"]["conversationId"]
            except Exception as e:
                raise HTTPException(500, f"Init conversation failed: {str(e)}")

    def _extract_context(self, messages: List[Message], device_id: str) -> tuple:
        """提取或生成会话上下文"""
        conv_id_pattern = re.compile(r"Conversation ID:\s*(\d+)")
        for msg in reversed(messages):
            if match := conv_id_pattern.search(msg.content):
                return (device_id, match.group(1))
        current_device_id = device_id or self.session_tools.gen_device_id()
        return (current_device_id, None)

    async def generate_response(
        self,
        messages: List[Message],
        stream_mode: bool,
        device_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """生成符合OpenAI格式的响应"""
        device_id, conv_id = self._extract_context(messages, device_id)
        
        if not conv_id:
            conv_id = await self._init_conversation(device_id)
            if not conv_id:
                raise HTTPException(500, "Failed to initialize conversation")

        payload = {
            "stream": True,
            "botCode": "AI_SEARCH",
            "userAction": "deep,online",
            "model": "deepseek",
            "conversationId": conv_id,
            "question": self._get_query(messages)
        }

        timestamp = str(int(time.time()))
        nonce = SessionTools.nanoid(21)
        sign = self.auth_tools.create_sign(timestamp, payload, nonce)

        # 生成统一响应ID和元数据
        response_id = f"chatcmpl-{uuid.uuid4()}"
        system_fingerprint = f"device:{device_id}|conv:{conv_id}"
        finish_sent = False  # 新增结束标记

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.API_BASE}/ai-search/chatApi/v1/chat",
                    json=payload,
                    headers={
                        **self.DEFAULT_HEADERS,
                        "deviceId": device_id,
                        "nonce": nonce,
                        "sign": sign,
                        "timestamp": timestamp
                    },
                    timeout=120
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        chunk = line[5:].strip()
                        if not chunk:
                            continue

                        try:
                            data = json.loads(chunk)
                            event_data = {
                                "id": response_id,
                                "created": int(time.time()),
                                "model": "deepseek",
                                "system_fingerprint": system_fingerprint,
                                "object": "chat.completion.chunk"
                            }

                            if data.get("type") == "follow_up" and not finish_sent:
                                # 发送符合SSE格式的最终块
                                yield f"data: {json.dumps({
                                    **event_data,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }]
                                })}\n\n"
                                finish_sent = True
                                continue

                            if content := data.get("content"):
                                # 发送符合SSE格式的内容块
                                yield f"data: {json.dumps({
                                    **event_data,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content}
                                    }]
                                })}\n\n"

                        except json.JSONDecodeError:
                            continue

                    # 确保发送结束标记
                    if not finish_sent:
                        yield f"data: {json.dumps({
                            "id": response_id,
                            "created": int(time.time()),
                            "model": "deepseek",
                            "system_fingerprint": system_fingerprint,
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        })}\n\n"

            except httpx.HTTPStatusError as e:
                error_msg = await response.aread()
                raise HTTPException(e.response.status_code, detail=error_msg.decode())
            except Exception as e:
                raise HTTPException(500, detail=str(e))

    def _get_query(self, messages: List[Message]) -> str:
        """获取最后用户消息"""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content.strip()
        raise HTTPException(400, "No valid user message found")

# ========== API端点 ==========
service = DeepSeekService()

@app.post("/v1/chat/completions")
async def chat_endpoint(
    request: ChatRequest,
    x_device_id: Optional[str] = Header(None)
):
    if request.stream:
        return StreamingResponse(
            service.generate_response(request.messages, True, x_device_id),
            media_type="text/event-stream"
        )

    # 处理非流式请求
    full_content = []
    async for chunk in service.generate_response(request.messages, False, x_device_id):
        try:
            data = json.loads(chunk)
            if data.get("choices") and data["choices"][0].get("delta"):
                if content := data["choices"][0]["delta"].get("content"):
                    full_content.append(content)
        except:
            continue

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "".join(full_content).split('Device ID')[0].strip()
            },
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
