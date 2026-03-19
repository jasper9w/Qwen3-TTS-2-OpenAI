# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import argparse
import base64
import io
import json
import os
import re
import shlex
import threading
import time
import uuid
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import Qwen3TTSModel


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _serialize_wav_base64(wav, sr: int, audio_format: str) -> str:
    fmt = (audio_format or "wav").strip().lower()
    if fmt != "wav":
        raise ValueError(f"Unsupported audio format: {audio_format}. Only wav is currently supported.")

    with io.BytesIO() as buf:
        sf.write(buf, wav, sr, format="WAV")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _audio_mime_type(audio_format: str) -> str:
    fmt = (audio_format or "wav").strip().lower()
    if fmt == "wav":
        return "audio/wav"
    raise ValueError(f"Unsupported audio format: {audio_format}. Only wav is currently supported.")


def _build_audio_tag_content(audio_b64: str, audio_format: str) -> str:
    mime_type = _audio_mime_type(audio_format)
    return (
        "[开始tts生成...]\n"
        "[音频生成中...]\n"
        "[生成完成!]\n"
        f"<audio src=\"data:{mime_type};base64,{audio_b64}\" controls></audio>"
    )


TRAILING_FLAG_BLOCK_RE = re.compile(
    r"(?P<body>.*?)(?:\s+(?P<flags>--[A-Za-z_][A-Za-z0-9_-]*=(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|\S+)"
    r"(?:\s+--[A-Za-z_][A-Za-z0-9_-]*=(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|\S+))*))\s*$",
    re.S,
)


FLAG_NAME_MAP = {
    "speaker": "speaker",
    "language": "language",
    "instruct": "instruct",
    "ref_text": "ref_text",
    "reference_text": "ref_text",
    "x_vector_only_mode": "x_vector_only_mode",
    "audio_format": "audio_format",
    "response_format": "audio_format",
    "voice": "voice",
}


def _coerce_flag_value(key: str, value: str) -> Any:
    value = value.strip()
    if key == "x_vector_only_mode":
        lowered = value.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return value


def _extract_trailing_cli_flags(text: str) -> Tuple[str, Dict[str, Any]]:
    text = (text or "").rstrip()
    if not text:
        return "", {}

    match = TRAILING_FLAG_BLOCK_RE.match(text)
    if not match:
        return text, {}

    body = match.group("body").rstrip()
    flags_blob = match.group("flags") or ""
    parsed: Dict[str, Any] = {}
    try:
        tokens = shlex.split(flags_blob)
    except ValueError:
        return text, {}

    for token in tokens:
        if not token.startswith("--") or "=" not in token:
            return text, {}
        raw_key, raw_value = token[2:].split("=", 1)
        key = FLAG_NAME_MAP.get(raw_key.strip().lower())
        if not key:
            continue
        parsed[key] = _coerce_flag_value(key, raw_value)

    return body, parsed


def _format_progress_line(stage: str, detail: str) -> str:
    return f"[{stage}] {detail}"


def _build_param_summary_text(params: Dict[str, Any]) -> str:
    ordered_keys = [
        "speaker",
        "language",
        "instruct",
        "ref_text",
        "x_vector_only_mode",
        "audio_format",
        "reference_audio",
        "tts_model_type",
        "sample_rate",
        "warnings",
    ]
    parts: List[str] = []
    for key in ordered_keys:
        if key not in params:
            continue
        value = params[key]
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = "|".join([str(x) for x in value])
        parts.append(f"{key}={value}")
    for key, value in params.items():
        if key in ordered_keys or value in (None, "", [], {}):
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _build_audio_tag(audio_b64: str, audio_format: str, info: Dict[str, Any]) -> str:
    mime_type = _audio_mime_type(audio_format)
    info_json = json.dumps(info, ensure_ascii=False, separators=(",", ":"))
    return (
        f"<audio src=\"data:{mime_type};base64,{audio_b64}\" controls "
        f"data-info=\"{escape(info_json, quote=True)}\"></audio>"
    )


def _extract_final_sse_content(full_content: str) -> str:
    lines = [line for line in (full_content or "").splitlines() if line != ""]
    if len(lines) >= 2:
        return "\n".join(lines[-2:])
    return full_content or ""


def _resolve_default_model() -> str:
    env_model = os.getenv("QWEN_TTS_MODEL")
    if env_model:
        return env_model

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice" / "snapshots"
    if cache_root.exists():
        snapshots = sorted([p for p in cache_root.iterdir() if p.is_dir()])
        if snapshots:
            return str(snapshots[-1])

    return "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def _normalize_base64_audio_ref(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reference audio in image_url.url must be a non-empty base64 string")

    value = value.strip()
    if value.startswith("data:"):
        header, sep, payload = value.partition(",")
        if not sep:
            raise ValueError("reference audio data URL is missing a comma separator")
        if ";base64" not in header.lower():
            raise ValueError("reference audio data URL must use base64 encoding")
        payload = "".join(payload.split())
        try:
            base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError("reference audio data URL does not contain valid base64 data") from exc
        return f"{header},{payload}"

    if "://" in value:
        raise ValueError("reference audio in image_url.url only supports base64 payloads, not URLs or file paths")

    payload = "".join(value.split())
    try:
        base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError("reference audio in image_url.url must be valid base64") from exc
    return payload


def _extract_image_audio_url(part: Dict[str, Any]) -> Optional[str]:
    for key in ("image_url", "input_image"):
        payload = part.get(key)
        if payload is None:
            continue
        if isinstance(payload, str):
            return _normalize_base64_audio_ref(payload)
        if isinstance(payload, dict):
            url = payload.get("url")
            if url:
                return _normalize_base64_audio_ref(url)
    return None


def _extract_message_text(content: Any) -> Tuple[str, Optional[str], Dict[str, Any]]:
    if isinstance(content, str):
        text, flags = _extract_trailing_cli_flags(content.strip())
        return text, None, flags

    if not isinstance(content, list):
        return "", None, {}

    text_parts: List[str] = []
    audio_ref: Optional[str] = None
    flags: Dict[str, Any] = {}
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in ("text", "input_text"):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        elif part_type in ("image_url", "input_image"):
            audio_ref = _extract_image_audio_url(part) or audio_ref

    merged_text = "\n".join(text_parts).strip()
    merged_text, flags = _extract_trailing_cli_flags(merged_text)
    return merged_text, audio_ref, flags


def _extract_prompt_payload(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    system_texts: List[str] = []
    user_texts: List[str] = []
    ref_audio: Optional[str] = None
    cli_params: Dict[str, Any] = {}

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text, audio_ref, extracted_flags = _extract_message_text(message.get("content"))
        if role == "system" and text:
            system_texts.append(text)
        elif role == "user":
            if text:
                user_texts.append(text)
            if audio_ref:
                ref_audio = audio_ref
            cli_params.update(extracted_flags)

    return {
        "text": user_texts[-1] if user_texts else None,
        "system_text": "\n".join(system_texts).strip() or None,
        "ref_audio": ref_audio,
        "cli_params": cli_params,
    }


@dataclass
class RuntimeConfig:
    default_model: str
    device: str
    dtype: torch.dtype
    flash_attn: bool
    request_timeout_seconds: float = 600.0


class TTSRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._models: Dict[str, Qwen3TTSModel] = {}
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load_model(self, model_name_or_path: str) -> Qwen3TTSModel:
        kwargs: Dict[str, Any] = {
            "device_map": self.config.device,
            "dtype": self.config.dtype,
        }
        if self.config.flash_attn and str(self.config.device).startswith("cuda"):
            kwargs["attn_implementation"] = "flash_attention_2"

        return Qwen3TTSModel.from_pretrained(model_name_or_path, **kwargs)

    def get_model(self, model_name_or_path: Optional[str]) -> Qwen3TTSModel:
        resolved = model_name_or_path or self.config.default_model
        with self._load_lock:
            if resolved not in self._models:
                self._models[resolved] = self._load_model(resolved)
            return self._models[resolved]

    def _token_count(self, model: Qwen3TTSModel, text: str) -> int:
        if not text:
            return 0
        try:
            encoded = model.processor(text=text, return_tensors="pt")
            return int(encoded["input_ids"].numel())
        except Exception:
            return 0

    def generate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        model_name = body.get("model") or self.config.default_model
        tts = self.get_model(model_name)

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="`messages` must be a non-empty list.")

        try:
            prompt = _extract_prompt_payload(messages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        text = (body.get("text") or prompt["text"] or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="No synthesis text found in request.")

        cli_params = prompt.get("cli_params", {})
        language = cli_params.get("language") or body.get("language") or "Auto"
        speaker = cli_params.get("speaker") or body.get("speaker") or cli_params.get("voice") or body.get("voice") or "Vivian"
        instruct = cli_params.get("instruct") or body.get("instruct") or prompt["system_text"]
        ref_audio = body.get("ref_audio") or prompt["ref_audio"]
        ref_text = cli_params.get("ref_text") or body.get("ref_text") or body.get("reference_text")
        audio_format = cli_params.get("audio_format") or body.get("audio_format") or body.get("response_format") or "wav"
        x_vector_only_mode = cli_params.get("x_vector_only_mode")
        if x_vector_only_mode is None:
            x_vector_only_mode = body.get("x_vector_only_mode")
        warnings: List[str] = []
        accepted_params: Dict[str, Any] = {
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode,
            "audio_format": audio_format,
            "reference_audio": "provided" if ref_audio else "none",
        }

        with self._infer_lock:
            if tts.model.tts_model_type == "custom_voice":
                if ref_audio:
                    warnings.append("reference audio was provided via image_url but ignored by custom_voice models")
                generate_kwargs = dict(text=text, language=language, speaker=speaker)
                if instruct:
                    generate_kwargs["instruct"] = instruct
                wavs, sr = tts.generate_custom_voice(**generate_kwargs)
            elif tts.model.tts_model_type == "base":
                if not ref_audio:
                    raise HTTPException(
                        status_code=400,
                        detail="Base voice-clone models require reference audio. Pass it through messages[].content[].image_url.url.",
                    )
                if x_vector_only_mode is None:
                    x_vector_only_mode = not bool(ref_text)
                    if x_vector_only_mode:
                        warnings.append("reference transcript missing; x_vector_only_mode was enabled automatically")
                wavs, sr = tts.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=bool(x_vector_only_mode),
                )
            elif tts.model.tts_model_type == "voice_design":
                if not instruct:
                    raise HTTPException(
                        status_code=400,
                        detail="VoiceDesign models require `instruct` or a system message with style instructions.",
                    )
                if ref_audio:
                    warnings.append("reference audio was provided via image_url but ignored by voice_design models")
                wavs, sr = tts.generate_voice_design(text=text, language=language, instruct=instruct)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported TTS model type: {tts.model.tts_model_type}")

        audio_b64 = _serialize_wav_base64(wavs[0], sr, audio_format=audio_format)
        tag_info = {
            "speaker": speaker if tts.model.tts_model_type == "custom_voice" else None,
            "language": language,
            "instruct": instruct,
            "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode,
            "audio_format": audio_format,
            "reference_audio": "provided" if ref_audio else "none",
            "tts_model_type": tts.model.tts_model_type,
            "sample_rate": sr,
            "warnings": warnings,
        }
        content_lines = [
            _format_progress_line("开始tts生成", "请求已接收"),
            _format_progress_line("参数解析完成", _build_param_summary_text(accepted_params)),
            _format_progress_line("音频生成中", f"tts_model_type={tts.model.tts_model_type}"),
            _format_progress_line("生成完成", f"sample_rate={sr}"),
            _build_audio_tag(audio_b64, audio_format=audio_format, info=tag_info),
        ]
        content_with_audio_tag = "\n".join(content_lines)
        prompt_tokens = self._token_count(tts, text)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": content_with_audio_tag,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 0,
                "total_tokens": prompt_tokens,
            },
        }


def _sse_encode(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode("utf-8")


def _chat_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if extra:
        chunk.update(extra)
    return chunk


async def _run_generate_with_timeout(runtime: TTSRuntime, body: Dict[str, Any]) -> Dict[str, Any]:
    timeout_seconds = float(runtime.config.request_timeout_seconds)
    try:
        return await asyncio.wait_for(asyncio.to_thread(runtime.generate, body), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"generation exceeded timeout of {int(timeout_seconds)} seconds",
        ) from exc


async def _chat_sse_stream(runtime: TTSRuntime, body: Dict[str, Any]) -> AsyncIterator[bytes]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = body.get("model") or runtime.config.default_model
    preview_text = ""
    preview_params = ""
    try:
        prompt = _extract_prompt_payload(body.get("messages") or [])
        preview_text = prompt.get("text") or ""
        cli_params = prompt.get("cli_params", {})
        preview_params = _build_param_summary_text(
            {
                "speaker": cli_params.get("speaker") or body.get("speaker") or cli_params.get("voice") or body.get("voice") or "Vivian",
                "language": cli_params.get("language") or body.get("language") or "Auto",
                "instruct": cli_params.get("instruct") or body.get("instruct") or prompt.get("system_text"),
                "ref_text": cli_params.get("ref_text") or body.get("ref_text") or body.get("reference_text"),
                "x_vector_only_mode": cli_params.get("x_vector_only_mode") if "x_vector_only_mode" in cli_params else body.get("x_vector_only_mode"),
                "audio_format": cli_params.get("audio_format") or body.get("audio_format") or body.get("response_format") or "wav",
                "reference_audio": "provided" if prompt.get("ref_audio") or body.get("ref_audio") else "none",
            }
        )
    except Exception:
        preview_text = ""
        preview_params = ""

    yield _sse_encode(json.dumps(_chat_chunk(
        chunk_id=chunk_id,
        created=created,
        model=model,
        delta={"role": "assistant"},
    ), ensure_ascii=False))
    yield _sse_encode(json.dumps(_chat_chunk(
        chunk_id=chunk_id,
        created=created,
        model=model,
        delta={"content": _format_progress_line("开始tts生成", "请求已接收") + "\n"},
    ), ensure_ascii=False))
    if preview_params:
        yield _sse_encode(json.dumps(_chat_chunk(
            chunk_id=chunk_id,
            created=created,
            model=model,
            delta={"content": _format_progress_line("参数解析完成", preview_params) + "\n"},
        ), ensure_ascii=False))
    if preview_text:
        yield _sse_encode(json.dumps(_chat_chunk(
            chunk_id=chunk_id,
            created=created,
            model=model,
            delta={"content": _format_progress_line("音频生成中", f"text_length={len(preview_text)}") + "\n"},
        ), ensure_ascii=False))

    try:
        result = await _run_generate_with_timeout(runtime, body)
    except HTTPException as exc:
        error_payload = {
            "error": {
                "message": exc.detail,
                "type": "timeout_error" if exc.status_code == 504 else "invalid_request_error",
                "code": "timeout" if exc.status_code == 504 else "bad_request",
            }
        }
        yield _sse_encode(json.dumps(error_payload, ensure_ascii=False))
        yield _sse_encode("[DONE]")
        return

    final_message = result["choices"][0]["message"]
    yield _sse_encode(json.dumps(_chat_chunk(
        chunk_id=chunk_id,
        created=created,
        model=result["model"],
        delta={
            "content": _extract_final_sse_content(final_message.get("content", "")),
        },
        finish_reason="stop",
    ), ensure_ascii=False))
    yield _sse_encode("[DONE]")


def create_app(config: RuntimeConfig) -> FastAPI:
    runtime = TTSRuntime(config)
    app = FastAPI(title="Qwen3-TTS OpenAI-Compatible API", version="0.1.0")
    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "default_model": config.default_model,
            "device": config.device,
            "dtype": str(config.dtype).replace("torch.", ""),
        }

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.default_model,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        if body.get("stream"):
            return StreamingResponse(
                _chat_sse_stream(runtime, body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return await _run_generate_with_timeout(runtime, body)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Dict[str, Any]:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(status_code=400, detail="`prompt` must be a non-empty string.")

        chat_body = {
            "model": body.get("model") or config.default_model,
            "messages": [{"role": "user", "content": prompt}],
            "language": body.get("language"),
            "speaker": body.get("speaker"),
            "voice": body.get("voice"),
            "instruct": body.get("instruct"),
            "audio_format": body.get("audio_format") or body.get("response_format"),
        }
        result = await _run_generate_with_timeout(runtime, chat_body)
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": result["created"],
            "model": result["model"],
            "choices": [
                {
                    "index": 0,
                    "text": result["choices"][0]["message"]["content"],
                    "finish_reason": "stop",
                }
            ],
            "usage": result["usage"],
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-openai-api",
        description="Serve Qwen3-TTS with an OpenAI-compatible /v1/chat/completions endpoint.",
    )
    parser.add_argument(
        "--model",
        default=_resolve_default_model(),
        help="Default model path or repo id. Defaults to local cached 0.6B-CustomVoice when available.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model loading, such as cpu or cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype used to load the model.",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 when running on CUDA.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds. Default is 600 seconds (10 minutes).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8001, help="Bind port.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = RuntimeConfig(
        default_model=args.model,
        device=args.device,
        dtype=_dtype_from_str(args.dtype),
        flash_attn=bool(args.flash_attn),
        request_timeout_seconds=float(args.request_timeout_seconds),
    )
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
