import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import lightning.pytorch as pl
import torch.nn.functional as F

from typing import Tuple
from diffusers import DDPMScheduler

from . import TimeEmbedding, PointNetEmb
from ..data.utils import GripperModel, compute_pose_from_vector


class ResBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256,
                 temp_emb_cat: bool = False):
        super(ResBlock, self).__init__()

        if temp_emb_cat:
            self.linear_layer = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        else:
            self.linear_layer = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
        self.hidden_dim = hidden_dim
        self.temp_emb_cat = temp_emb_cat

    def forward(self, x, temp_emb, pcd_emb, text_emb):
        if self.temp_emb_cat:
            x_expand = torch.cat((x, temp_emb, pcd_emb, text_emb), dim=1)
        else:
            x_expand = torch.cat((x, pcd_emb, text_emb), dim=1) * temp_emb
        x = x + self.linear_layer(x_expand)
        return x

class DiffDexGrasp(pl.LightningModule):
    def __init__(self,
                 diffusion_step: int,
                 noise_scheduler: DDPMScheduler,
                 gripper: GripperModel,
                 optim: str = 'Adam',
                 lr: float = 1e-3,
                 input_dim: int = 32,
                 hidden_dim: int = 256,
                 res_layers_num: int = 4,
                 temp_emb_cat: bool = False,
                 pcd_loss_scale = 1.0):
        super(DiffDexGrasp, self).__init__()

        ## Build Network
        self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
        )

        if temp_emb_cat:
            self.temp_emb = TimeEmbedding(diffusion_step, hidden_dim // 4, hidden_dim)
        else:
            self.temp_emb = TimeEmbedding(diffusion_step, hidden_dim // 4, hidden_dim * 3)
        self.linears = nn.ModuleList([ResBlock(hidden_dim, temp_emb_cat=temp_emb_cat) for _ in range(res_layers_num)])
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.pcd_emb = PointNetEmb(emb_dim=hidden_dim)

        self.text_compress = nn.Sequential(
                nn.Linear(1536, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.temp_emb_cat = temp_emb_cat
        self.initialize()

        self.optim=optim
        self.lr = lr
        self.gripper = gripper
        self.pcd_loss_scale = pcd_loss_scale
        self.diffusion_step = diffusion_step
        self.noise_scheduler = noise_scheduler

        self.train_output = {'loss': []}

    def initialize(self):
        for module in self.linears:
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                init.zeros_(module.bias)
        for module in self.input_layer:
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                init.zeros_(module.bias)
        init.kaiming_uniform_(self.output_layer.weight)
        init.zeros_(self.output_layer.bias)

    def forward(self,
                noisy_x: torch.Tensor,
                timesteps: torch.Tensor,
                obj_pcd: torch.Tensor,
                hand_pcd: torch.Tensor,
                text_emb: torch.Tensor) -> torch.Tensor:
        """
        :param noisy_x: (B, D)
        :param timesteps: (B,)
        :param obj_pcd: (B, N1, 3)
        :param hand_pcd: (B, N2, 3)
        :param text_emb: (B, 1536)
        :return: (B, D)
        """
        t_emb = self.temp_emb(timesteps)
        pcd_emb = self.pcd_emb(obj_pcd, hand_pcd)
        text_compressed_emb = self.text_compress(text_emb)
        x = noisy_x
        x = self.input_layer(x)
        for idx, layer in enumerate(self.linears):
            x = layer(x, t_emb, pcd_emb, text_compressed_emb)
        pred_noise = self.output_layer(x)
        return pred_noise

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        pose, joint, obj_pcd, text_emb = batch
        pose = pose.to(torch.float32)
        joint = joint.to(torch.float32)
        obj_pcd = obj_pcd.to(torch.float32)
        text_emb = text_emb.to(torch.float32)


        pose_a = pose[:, :3, 0]
        pose_b = pose[:, :3, 1]
        pose_t = pose[:, :3, 3]
        X = torch.cat((pose_a, pose_b, pose_t, joint), dim=1).to(torch.float)
        noise = torch.randn(X.shape).to(X.device)
        bs = X.shape[0]

        timesteps = torch.randint(
            0, self.diffusion_step, (bs,),
        ).to(X.device)

        noisy_x = self.noise_scheduler.add_noise(X, noise, timesteps)
        noisy_pose, noisy_joint = compute_pose_from_vector(noisy_x)

        hand_pcd, _, _ = self.gripper.compute_pcd(noisy_pose, noisy_joint)

        noise_pred = self.forward(noisy_x, timesteps, obj_pcd, hand_pcd, text_emb)

        loss = torch.mean(torch.square(noise_pred - noise))
        noisy_x_pred = self.noise_scheduler.add_noise(X, noise_pred, timesteps)
        noisy_pose_pred, noisy_joint_pred = compute_pose_from_vector(noisy_x_pred)
        hand_pcd_pred, _, _ = self.gripper.compute_pcd(noisy_pose_pred, noisy_joint_pred)
        loss += self.pcd_loss_scale * F.mse_loss(hand_pcd_pred, hand_pcd)

        self.log('train_loss', loss)
        self.train_output['loss'].append(loss.item())
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pose, joint, obj_pcd, text_emb = batch
        pose = pose.to(torch.float32)
        joint = joint.to(torch.float32)
        obj_pcd = obj_pcd.to(torch.float32)
        text_emb = text_emb.to(torch.float32)

        pose_a = pose[:, :3, 0]
        pose_b = pose[:, :3, 1]
        pose_t = pose[:, :3, 3]
        X = torch.cat((pose_a, pose_b, pose_t, joint), dim=1).to(torch.float)
        noise = torch.randn(X.shape).to(X.device)
        bs = X.shape[0]

        timesteps = torch.randint(
            0, self.diffusion_step, (bs,),
        ).to(X.device)

        noisy_x = self.noise_scheduler.add_noise(X, noise, timesteps)
        noisy_pose, noisy_joint = compute_pose_from_vector(noisy_x)

        hand_pcd, _, _ = self.gripper.compute_pcd(noisy_pose, noisy_joint)

        noise_pred = self.forward(noisy_x, timesteps, obj_pcd, hand_pcd, text_emb)

        loss = F.mse_loss(noise_pred, noise)
        noisy_x_pred = self.noise_scheduler.add_noise(X, noise_pred, timesteps)
        noisy_pose_pred, noisy_joint_pred = compute_pose_from_vector(noisy_x_pred)
        hand_pcd_pred, _, _ = self.gripper.compute_pcd(noisy_pose_pred, noisy_joint_pred)
        loss += self.pcd_loss_scale * F.mse_loss(hand_pcd_pred, hand_pcd)

        self.log('val_loss', loss)

    def on_train_epoch_end(self) -> None:
        print('training_epoch_end')
        self.noise_scheduler.save_pretrained(self.logger.log_dir)
        self.log('train_epoch_loss', np.asarray(self.train_output['loss']).mean())
        self.train_output['loss'].clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optim == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError