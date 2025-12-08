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

import os
from typing import Any

"""Text processing utilities."""

SYSTEM_PROMPT = "You are a helpful assistant."
"""Default system prompt."""

REASONING_PROMPT = """Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""
"""Reasoning addon prompt."""


def create_conversation(
    *,
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    response: str = "",
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
    vision_kwargs: dict | None = None,
) -> list[dict]:
    """Create chat conversation for transformers.

    Args:
        user_prompt: User prompt.
        system_prompt: System prompt.
        response: Assistant response.
        images: List of images.
        videos: List of videos.
        vision_kwargs: Keyword arguments for vision processor (see `cosmos_reason1_utils.vision.VisionConfig`).

    Returns:
        conversation: Chat conversation.
    """
    user_content = []
    if images is not None:
        for image in images:
            user_content.append({"type": "image", "image": image})
    if videos is not None:
        for video in videos:
            user_content.append({"type": "video", "video": video})
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_content})
    if response:
        conversation.append({"role": "assistant", "content": response})
    if vision_kwargs:
        set_vision_kwargs(conversation, vision_kwargs)
    return conversation


def create_conversation_openai(
    *,
    user_prompt: str = "",
    response: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
) -> list[dict]:
    """Create chat conversation for OpenAI API.

    Specification: https://platform.openai.com/docs/api-reference/messages

    Args:
        system_prompt: System prompt.
        user_prompt: User prompt.
        response: Assistant response.
        images: List of images.
        videos: List of videos.

    Returns:
        conversation: Chat conversation.
    """
    user_content = []
    if images is not None:
        for image in images:
            user_content.append(
                {"type": "image_url", "image_url": {"url": _get_media_url(image)}}
            )
    if videos is not None:
        for video in videos:
            if isinstance(video, dict):
                user_content.append({"type": "video", "video": video["frame_list"]})
            else:
                user_content.append(
                    {"type": "video_url", "video_url": {"url": _get_media_url(video)}}
                )
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": user_content})
    if response:
        conversation.append({"role": "assistant", "content": response})
    return conversation


def _get_media_url(path: str) -> str:
    if ":" in path:
        return path
    path = os.path.abspath(path)
    return f"file://{path}"


def set_vision_kwargs(conversation: list[dict], vision_kwargs: dict):
    """Set vision kwargs for all media messages in conversation.

    Args:
        conversation: Conversation (see `create_conversation`).
        vision_kwargs: Keyword arguments for vision processor (see `cosmos_reason1_utils.vision.VisionConfig`).
    """
    for msg in conversation:
        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        for msg in content:
            if isinstance(msg, dict) and msg.get("type", None) in [
                "image",
                "video",
                "image_url",
                "video_url",
            ]:
                msg |= vision_kwargs
