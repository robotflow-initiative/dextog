import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hydra
import torch
import os.path as osp
import lightning.pytorch as pl

from omegaconf import DictConfig
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from data.utils import GripperModel
from model.diffdexgrasp import DiffDexGrasp
from data.dataset import ObjectDataset, TextDataset, GraspDataset

@hydra.main(version_base="v1.2", config_path='conf', config_name='train')
def train(cfg: DictConfig) -> None:
    dataset = GraspDataset(osp.join('assets', 'grasp', 'TOG'),
                           object_dataset=ObjectDataset(osp.join('assets', 'object', 'TOG')),
                           text_dataset=TextDataset(),
                           data_augmentation=cfg.train.data_aug)

    gripper = GripperModel('shadow', use_complete_points=cfg.train.complete_hand)

    diffusion_step = cfg.diffusion.step

    noise_scheduler = DDPMScheduler(num_train_timesteps=diffusion_step)

    model = DiffDexGrasp(diffusion_step=diffusion_step,
                         noise_scheduler=noise_scheduler,
                         gripper=gripper,
                         lr=cfg.train.lr,
                         pcd_loss_scale= cfg.train.pcd_loss_coeff,
                         **cfg.model)
    logger = pl.loggers.TensorBoardLogger("tb_logs", **cfg.logger)
    if cfg.train.use_val_set:
        checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor='val_loss', mode='min')
        trainer = pl.Trainer(accelerator='gpu', logger=logger, callbacks=[checkpoint_callback], max_epochs=10000)
        dataset_size = len(dataset)
        val_size = int(dataset_size * cfg.train.val_ratio)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.bs, shuffle=True, num_workers=24)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.bs, shuffle=False, num_workers=24)
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor='train_loss', mode='min')
        trainer = pl.Trainer(accelerator='gpu', logger=logger, max_epochs=10000, callbacks=[checkpoint_callback])
        train_dataset = dataset
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.bs, shuffle=True, num_workers=24)
        trainer.fit(model=model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    train()
