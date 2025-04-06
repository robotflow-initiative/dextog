import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hydra

from loguru import logger
from omegaconf import DictConfig

from train.trainer import Trainer

@hydra.main(version_base="v1.2", config_path='conf', config_name='train')
def main(cfg: DictConfig):
    task = cfg.task
    logger.info(f"The ppo training for {task} begins")
    trainer = Trainer(cfg)
    logger.info(f"The ppo checkpoint for {task} is saved at {trainer.train()}")

if __name__ == "__main__":
    main()