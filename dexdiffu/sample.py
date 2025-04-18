import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import hydra
import torch
import os.path as osp

sys.path.append('.')
torch.set_default_dtype(torch.float)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from omegaconf import DictConfig
from diffusers import DDPMScheduler

from .metrics import cal_q1, TTA, cal_pen
from .model.diffdexgrasp import DiffDexGrasp
from .data.dataset import ObjectDataset, TextDataset
from .data.utils import GripperModel, visualize_gripper_and_object, compute_pose_from_vector

@hydra.main(version_base="v1.2", config_path='conf', config_name='sample')
def sample(cfg: DictConfig) -> None:
    diffusion_step = cfg.diffusion.step
    object_dataset = ObjectDataset(osp.join('assets', 'object', 'TOG'), subdir=cfg.sample.category)
    text_dataset = TextDataset()
    gripper = GripperModel('shadow', use_complete_points=cfg.sample.complete_hand)

    scheduler = DDPMScheduler.from_pretrained(cfg.sample.scheduler_path)
    scheduler.set_timesteps(diffusion_step)

    batch_size = 8
    sample_size = (batch_size, 32)

    model = DiffDexGrasp.load_from_checkpoint(cfg.sample.ckpt_path,
                                             diffusion_step=diffusion_step,
                                             noise_scheduler=scheduler,
                                             gripper=gripper,
                                             **cfg.model)
    model.eval()

    noise = torch.randn(sample_size).to(device)

    for idx in range(len(object_dataset)):
        input = noise.to(device).contiguous()
        catagory = object_dataset.data[idx].split(os.sep)[0]
        text_emb = text_dataset.fetch_text_emb(catagory, True)
        obj_pcd = torch.from_numpy(object_dataset[idx]).repeat((batch_size, 1, 1)).to(device).to(torch.float32)

        text_emb = torch.from_numpy(text_emb).repeat((batch_size, 1)).to(device).to(torch.float32)
        for t in scheduler.timesteps:
            with torch.no_grad():
                new_pose, new_joint = compute_pose_from_vector(input)
                hand_pcd, _, _ = gripper.compute_pcd(new_pose, new_joint)
                noisy_residual = model(input, t.repeat((batch_size)).to(device), obj_pcd, hand_pcd, text_emb)
            previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            input = previous_noisy_sample
        vertices, faces = object_dataset.fetch_mesh(object_dataset.data[idx])
        Tbase, joints_state = compute_pose_from_vector(input)
        q1_ = cal_q1(gripper, Tbase, joints_state, vertices, faces)
        depth_ = cal_pen(gripper, Tbase, joints_state, vertices, faces)
        pre_t = time.time()
        Tbase, joints_state = TTA(gripper, Tbase, joints_state, vertices, faces)
        q1 = cal_q1(gripper, Tbase, joints_state, vertices, faces)
        print("time cost: ", time.time() - pre_t)
        depth = cal_pen(gripper, Tbase, joints_state, vertices, faces)
        metrics = {'q1_pre': q1_, 'q1': q1, 'depth_pre': depth_, 'depth': depth}
        visualize_gripper_and_object(gripper, Tbase, joints_state, obj_pcd, metrics)

if __name__ == '__main__':
    sample()


