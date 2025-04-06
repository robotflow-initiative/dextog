import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hydra
import numpy as np

from typing import Dict
from loguru import logger
from omegaconf import DictConfig

from train.trainer import Trainer

def logging_info(success_dict: Dict, task: str, total_success_rate: float):
    for key, value in success_dict.items():
        success_rate = np.sum(value) * 1.0 / len(value)
        logger.info(f"success rate of {key} is: {success_rate}")
    logger.info(f"success rate of {task} is: {total_success_rate}")

@hydra.main(version_base="v1.2", config_path='conf', config_name='test')
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    cfg.train.ckpt = cfg.policy_path.get(cfg.task)
    success_dict, success_rate = trainer.run()
    logging_info(success_dict, cfg.task, success_rate)

if __name__ == "__main__":
    main()