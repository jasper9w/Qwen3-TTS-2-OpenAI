# coding=utf-8

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_tts import Qwen3TTSModel


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate one sample wav with Qwen3-TTS-12Hz-0.6B-CustomVoice.")
    parser.add_argument("--model", default=_resolve_default_model(), help="Model path or repo id.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0.")
    parser.add_argument("--speaker", default="Vivian", help="CustomVoice speaker name.")
    parser.add_argument("--language", default="Chinese", help="Target language.")
    parser.add_argument(
        "--text",
        default="你好，这是一段使用 Qwen3-TTS-12Hz-0.6B-CustomVoice 生成的示例音频。",
        help="Text to synthesize.",
    )
    parser.add_argument("--output", default="output/qwen3_tts_0_6b_customvoice_sample.wav", help="Output wav path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        **({"attn_implementation": "flash_attention_2"} if args.device.startswith("cuda") else {}),
    )
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        max_new_tokens=512,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wavs[0], sr)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
