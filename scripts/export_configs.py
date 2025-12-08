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

"""Export config defaults and schemas."""

import argparse
import json
import pathlib

import pydantic
import toml
import vllm
import yaml
from cosmos_reason2_utils.script.inference import InputConfig as InferenceConfig
from cosmos_rl.policy.config import Config as CosmosRlConfig

ROOT_DIR = pathlib.Path(__file__).parents[1].absolute()


class SamplingParams(vllm.SamplingParams, omit_defaults=False): ...


def pydantic_to_yaml(default: pydantic.BaseModel, schema_path: str) -> str:
    return "\n".join(
        [
            f"$schema: ./{schema_path}",
            yaml.dump(default.model_dump()),
        ]
    )


def pydantic_to_toml(default: pydantic.BaseModel, schema_path: str) -> str:
    return "\n".join(
        [
            f"#:schema ./{schema_path}",
            "",
            toml.dumps(default.model_dump()),
        ]
    )


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=f"{ROOT_DIR}/configs",
        help="Output directory",
    )
    args = args.parse_args()

    output_dir: pathlib.Path = args.output.absolute()
    schemas_dir = output_dir / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)

    inference_config = InferenceConfig()
    (output_dir / "inference_config.yaml").write_text(
        pydantic_to_yaml(inference_config, "schemas/inference_config.json")
    )
    (output_dir / "schemas/inference_config.json").write_text(
        json.dumps(inference_config.model_json_schema(), indent=2)
    )

    cosmos_rl_config = CosmosRlConfig()
    (output_dir / "cosmos_rl_config.toml").write_text(
        pydantic_to_toml(cosmos_rl_config, "schemas/cosmos_rl_config.toml")
    )
    (output_dir / "schemas/cosmos_rl_config.json").write_text(
        json.dumps(cosmos_rl_config.model_json_schema(), indent=2)
    )


if __name__ == "__main__":
    main()
