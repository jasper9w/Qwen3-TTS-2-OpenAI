# Qwen3-TTS OpenAI-Compatible FastAPI

这个仓库新增了一个 OpenAI 兼容的 FastAPI 服务入口。

## 启动手顺

1. 进入仓库根目录。

```bash
cd /Users/wei/works/github/Qwen3-TTS
```

2. 确认 `uv` active 环境已经可用。

```bash
uv --version
```

3. 如果模型已经缓存到本地，直接启动脚本。

```bash
bash scripts/start_openai_api.sh
```

默认配置：

- 模型：优先使用本地缓存的 `Qwen3-TTS-12Hz-0.6B-CustomVoice` snapshot；如果没找到，则回退到 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- 地址：`0.0.0.0:8001`
- 设备：`cpu`
- 精度：`float32`
- 单请求超时：`600` 秒，也就是 `10` 分钟

4. 如果你要改模型或端口，可以通过环境变量覆盖。

```bash
MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/<snapshot_id>" \
HOST=127.0.0.1 \
PORT=8002 \
DEVICE=cpu \
DTYPE=float32 \
REQUEST_TIMEOUT_SECONDS=600 \
bash scripts/start_openai_api.sh
```

5. 用健康检查确认服务已启动。

```bash
curl http://127.0.0.1:8001/health
```

6. 发送一个最小 `chat/completions` 请求。

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "你好，这是一条启动后的联调请求。 --speaker=Vivian --language=Chinese"}
        ]
      }
    ]
  }'
```

## 启动脚本

脚本路径：

- `scripts/start_openai_api.sh`

脚本默认使用：

- `uv run --active --no-sync`
- `PYTHONPATH=<repo_root>`
- `UV_CACHE_DIR=/tmp/uv-cache`
- `QWEN_TTS_DISABLE_MLX=1`
- `REQUEST_TIMEOUT_SECONDS=600`

如果你已经用 `uv` 准备好了环境，建议直接在仓库根目录运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --active --no-sync python -m qwen_tts.openai_api \
  --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/<snapshot_id> \
  --device cpu \
  --dtype float32 \
  --port 8001
```

如果项目已经安装成脚本入口，也可以直接执行：

```bash
qwen-tts-openai-api --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/<snapshot_id> --device cpu --dtype float32 --port 8001
```

在 macOS 上，如果全局环境里装了 `mlx`，仓库现在会默认禁用它，避免加载模型时误走 MLX/Metal 分支导致进程崩溃。如需恢复，可以显式设置 `QWEN_TTS_DISABLE_MLX=0`。

主要兼容的是 `POST /v1/chat/completions`。

参数输入约定：

- 把业务参数写在 `user` 文本最后，使用后缀形式
- 例如：`你好 --speaker=Vivian --language=Chinese`
- 当前支持：`--speaker=...`、`--language=...`、`--instruct=...`、`--ref_text=...`、`--x_vector_only_mode=true|false`、`--audio_format=wav`

为了在 OpenAI 消息结构里携带参考音，服务会把 `image_url.url` 当作音频引用来解析，支持：

- `data:audio/wav;base64,...`
- 裸 base64 音频字符串

不支持：

- `http(s)://...`
- `file:///...`
- 本地路径

## 1. CustomVoice

`Qwen3-TTS-12Hz-0.6B-CustomVoice` 本身不使用参考音，因此即使传了 `image_url` 也只会在校验通过后被忽略。这里如果传了，就必须仍然是 base64。

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "你好，这是一段 OpenAI chat completions 风格的语音合成请求。 --speaker=Vivian --language=Chinese"}
        ]
      }
    ]
  }'
```

返回体里，主要读取 `choices[0].message.content`。它会直接返回一段可渲染的 HTML 风格文本，格式类似：

```html
[开始tts生成] 请求已接收
[参数解析完成] speaker=Vivian language=Chinese audio_format=wav reference_audio=none
[音频生成中] tts_model_type=custom_voice
[生成完成] sample_rate=24000
<audio src="data:audio/wav;base64,..." controls data-info="{...}"></audio>
```

不会再额外返回 `message.audio`、`warnings`、`service_meta` 这些扩展字段。

## 2. Base Voice Clone

如果把模型切成 `Qwen3-TTS-12Hz-0.6B-Base`，则 `image_url.url` 会被当作参考音使用。这里的 `url` 实际只接受 base64：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请用参考音的音色读这句话。 --language=Chinese --ref_text=\"这是参考音对应的文本。\""},
          {"type": "image_url", "image_url": {"url": "data:audio/wav;base64,<BASE64_AUDIO>"}}
        ]
      }
    ]
  }'
```

如果没传 `ref_text`，服务会自动退化到 `x_vector_only_mode=true`。

## 3. SSE Streaming

如果请求里传 `stream: true`，服务会返回 `text/event-stream`，也就是 SSE。

```bash
curl -N http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "你好，这是一个 SSE 测试请求。 --speaker=Vivian --language=Chinese"}
        ]
      }
    ]
  }'
```

这个流会按阶段返回：

- `role=assistant`
- `[开始tts生成] 请求已接收`
- `[参数解析完成] ...`
- `[音频生成中] ...`
- 最终 `[生成完成] ...` 加 `<audio ...>` 标签
- `data: [DONE]`

## 4. Legacy Completions

也提供了 `POST /v1/completions` 包装，但这个接口本身不适合传 `image_url`，只适合纯文本 TTS：

```bash
curl http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "prompt": "你好，这是 legacy completions 接口。 --speaker=Vivian --language=Chinese"
  }'
```
