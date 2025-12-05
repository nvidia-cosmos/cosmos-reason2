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

ARG BASE_IMAGE=nvcr.io/nvidia/vllm:25.11-py3

FROM ${BASE_IMAGE}

# Set the DEBIAN_FRONTEND environment variable to avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git-lfs \
        gpg \
        lsb-release \
        tree \
        wget

# Install redis: https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-on-linux/#install-on-ubuntudebian
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    chmod 644 /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends redis

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=docker/nightly-requirements.txt,target=requirements.txt \
    --mount=type=bind,source=cosmos_reason2_utils,target=cosmos_reason2_utils \
    uv pip install --system --break-system-packages -r requirements.txt

ENTRYPOINT ["/bin/bash"]
