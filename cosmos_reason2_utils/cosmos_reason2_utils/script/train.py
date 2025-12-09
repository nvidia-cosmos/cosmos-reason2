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

import sys
import tempfile
from typing import Annotated

import pydantic
import toml
import tyro
from cosmos_rl.policy.config import Config
from tyro.conf import Positional

# from cosmos_reason2_utils.train import CustomConfig


class BaseArgs(pydantic.BaseModel):
    config_file: pydantic.FilePath | None = None
    """Path to config toml file."""


class Args(pydantic.BaseModel):
    script: Positional[pydantic.FilePath] = tyro.MISSING
    """Script to run."""

    config: Annotated[Config, tyro.conf.arg(prefix_name=False)] = Config()
    """Config."""
    # custom: CustomConfig = CustomConfig()
    # """Custom config."""
    log_dir: str | None = None
    """Log file directory."""


def main():
    base_args, unknown_args = tyro.cli(
        BaseArgs, return_unknown_args=True, add_help=False
    )
    if base_args.config_file is not None:
        config = Config.model_validate(toml.loads(open(base_args.config_file).read()))
    else:
        config = Config()
    # custom_config =CustomConfig.model_validate(config.custom)
    args = tyro.cli(
        Args,
        description=__doc__,
        args=unknown_args,
        default=Args(
            config=config,
            # custom=custom_config,
        ),
        config=(tyro.conf.AvoidSubcommands,),
    )

    # Save to config toml
    config_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".toml")
    config_filename = config_file.name
    config_kwargs = args.config.model_dump()
    config_kwargs["custom"] = args.custom.model_dump()
    config_file.write(toml.dumps(config_kwargs))
    config_file.close()
    print(f"Config file: {config_filename}")

    # Launch the training job
    sys.argv[:] = [
        "cosmos-rl",
        "--config",
        f"{config_filename}",
        "--log-dir",
        f"{args.log_dir}",
        f"{args.script}",
    ]
    print("Command: " + " ".join(sys.argv))

    # HACK
    return

    import cosmos_rl.launcher.launch_all

    cosmos_rl.launcher.launch_all.main()


if __name__ == "__main__":
    main()
