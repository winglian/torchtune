# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys

import torch

from omegaconf import DictConfig
from torchtune import config, utils


@config.parse
def main(cfg: DictConfig) -> None:
    checkpointer = config.instantiate(cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()

    with utils.set_default_dtype(torch.bfloat16), "cuda":
        model = config.instantiate(cfg.model)

    model.load_state_dict(ckpt_dict[utils.MODEL_KEY])


if __name__ == "__main__":
    sys.exit(main())
