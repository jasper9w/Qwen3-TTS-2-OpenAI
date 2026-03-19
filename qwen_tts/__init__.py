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

"""
qwen_tts: Qwen-TTS package.
"""

import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

try:
    import transformers.utils.import_utils as _transformers_import_utils

    # Qwen3-TTS does not use MLX in this repo, but a globally installed `mlx`
    # package can be auto-detected by Transformers and later imported on macOS.
    # On machines without a usable Metal device this can crash the process during
    # model loading, so we disable that optional backend by default.
    if os.environ.get("QWEN_TTS_DISABLE_MLX", "1") == "1":
        _transformers_import_utils._mlx_available = False
except Exception:
    pass

from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["__version__"]
