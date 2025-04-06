import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hydra
import os.path as osp
from loguru import logger

from omegaconf import DictConfig

from utils import filter_grasp_pose, augment_dataset, filter_not_catch

@hydra.main(version_base="v1.2", config_path='conf', config_name='refine_loop')
def main(cfg: DictConfig):
    rl_output_path = cfg.rl_output_path
    diffu_output_path = cfg.diffu_output_path
    rl_dataset_path = cfg.init_dataset_path
    logger.add(osp.join("assets", "tmp", f"{cfg.task}.log"))
    for i in range(cfg.iter_max_time):
        rl_output_path_ = rl_output_path + f"_{i}"
        diffu_output_path_ = diffu_output_path + f"_{i}"
        logger.info(f"rl output path: {rl_output_path_}")
        logger.info(f"diffu output path: {diffu_output_path_}")
        logger.info(f"rl dataset path: {rl_dataset_path}")
        _, success_rate = filter_grasp_pose(cfg.rl_cfg, rl_dataset_path, rl_output_path_)
        logger.info(f"The success rate of {cfg.task} after {i}th iter is {success_rate}")
        augment_dataset(cfg.diffu_cfg, rl_output_path_, diffu_output_path_, iter_time=i)
        logger.info(f"The Diffusion Process End")
        cfg.diffu_cfg.train_cfg.train.max_iter = cfg.diffu_cfg.train_cfg.train.max_iter + 1500
        filter_not_catch(diffu_output_path_, cfg.task)
        rl_dataset_path = diffu_output_path_



if __name__ == "__main__":
    main()