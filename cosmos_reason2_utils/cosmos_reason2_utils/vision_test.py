# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import torch

from cosmos_reason2_utils.vision import save_tensor

_FRAMES = 2
_CHANNELS = 3
_WIDTH = 4
_HEIGHT = 5


def test_save_tensor(tmp_path: Path):
    save_tensor(torch.ones((_CHANNELS, _WIDTH, _HEIGHT)), f"{tmp_path}/image")
    assert (tmp_path / "image/0.png").exists()
    save_tensor(torch.ones((_FRAMES, _CHANNELS, _WIDTH, _HEIGHT)), f"{tmp_path}/video")
    for i in range(_FRAMES):
        assert (tmp_path / f"video/{i}.png").exists()
