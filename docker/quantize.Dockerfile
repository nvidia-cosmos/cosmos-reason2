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

FROM vllm/vllm-openai:v0.11.0

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    git+https://github.com/vllm-project/llm-compressor.git@6e459ed \
    git+https://github.com/huggingface/transformers.git@def9a7ef057b13d04aeeaa150e3ce63afa151d4e \
    qwen_vl_utils==0.0.14

ENTRYPOINT ["/bin/bash"]
