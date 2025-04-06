import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import hydra
import torch
import numpy as np
import os.path as osp

sys.path.append('.')
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm import trange
from omegaconf import DictConfig
from diffusers import DDPMScheduler

from metrics import TTA
from model.diffdexgrasp import DiffDexGrasp
from data.dataset import ObjectDataset, TextDataset
from data.utils import GripperModel, compute_pose_from_vector

@hydra.main(version_base="v1.2", config_path='conf', config_name='generate')
def generate(cfg: DictConfig) -> None:
    diffusion_step = cfg.diffusion.step
    object_dataset = ObjectDataset(osp.join('assets', 'object', 'TOG'))
    text_dataset = TextDataset()
    gripper = GripperModel('shadow', use_complete_points=cfg.generate.complete_hand)

    scheduler = DDPMScheduler.from_pretrained(cfg.generate.scheduler_path)
    scheduler.set_timesteps(diffusion_step)

    batch_size = cfg.generate.bs
    sample_size = (batch_size, 32)
    save_path = cfg.generate.save_path

    model = DiffDexGrasp.load_from_checkpoint(cfg.generate.ckpt_path,
                                              diffusion_step=diffusion_step,
                                              noise_scheduler=scheduler,
                                              gripper=gripper,
                                              **cfg.model)
    model.eval()

    noise = torch.randn(sample_size).to(device)
    for idx in range(len(object_dataset)):
        print(f'start to generate grasp of {idx}th object')
        obj_path = object_dataset.data[idx]
        obj_path = osp.join(save_path, obj_path)
        vertices, faces = object_dataset.fetch_mesh(object_dataset.data[idx])
        if not osp.exists(obj_path):
            os.makedirs(obj_path)
        output_pose = []
        output_joint = []
        for _ in trange(cfg.generate.total_num // batch_size):
            input = noise.to(device).contiguous()
            category = object_dataset.data[idx].split(os.sep)[0]
            text_emb = text_dataset.fetch_text_emb(category, True)
            obj_pcd = torch.from_numpy(object_dataset[idx]).repeat((batch_size, 1, 1)).to(device).to(torch.float32)
            text_emb = torch.from_numpy(text_emb).repeat((batch_size, 1)).to(device).to(torch.float32)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    new_pose, new_joint = compute_pose_from_vector(input)
                    hand_pcd, _, _ = gripper.compute_pcd(new_pose, new_joint)
                    noisy_residual = model(input, t.repeat((batch_size)).to(device), obj_pcd, hand_pcd, text_emb)
                previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                input = previous_noisy_sample
            Tbase, joints_state = compute_pose_from_vector(input)
            if cfg.generate.tta:
                Tbase, joints_state = TTA(gripper, Tbase, joints_state, vertices, faces)
            output_pose.append(Tbase.detach().cpu().numpy())
            output_joint.append(joints_state.detach().cpu().numpy())
        output_pose = np.concatenate(output_pose)
        output_joint = np.concatenate(output_joint)
        np.save(osp.join(obj_path, 'pose.npy'), output_pose)
        np.save(osp.join(obj_path, 'joint_state.npy'), output_joint)

if __name__ == '__main__':
    generate()


