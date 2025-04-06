import os
import json
import torch
import numpy as np
import open3d as o3d
import os.path as osp

from tqdm import tqdm
from typing import Tuple
from kaolin.io.off import import_mesh
from torch.utils.data import Dataset, DataLoader

from metrics import cal_pen, cal_q1
from data.utils import normalized_joint_state, recover_joint_state, GripperModel, visualize_point_cloud

class ObjectDataset():
    def __init__(self,
                 path: str = osp.join('assets', 'object', 'TOG'),
                 point_num: int = 2048,
                 subdir: str = None):

        data = []
        if subdir is None:
            dirs = os.listdir(path)
            subdirs = [dir for dir in dirs if osp.isdir(osp.join(path, dir))]
        else:
            subdirs = [subdir]
        for dir in subdirs:
            subpath = osp.join(path, dir)
            obj_paths = os.listdir(subpath)
            for obj_path in obj_paths:
                data.append(osp.join(dir, obj_path))

        self.data = data
        self.rootpath = path
        self.point_num = point_num

    def __getitem__(self, idx) -> np.ndarray:
        return self.fetch_pointcloud(self.data[idx])

    def __len__(self):
        return len(self.data)

    def fetch_pointcloud(self, objpath: str) -> np.ndarray:
        objfile = osp.join(self.rootpath, objpath, 'part_meshes', 'complete.ply')
        pcd = o3d.io.read_point_cloud(objfile)
        pcd = np.asarray(pcd.points)

        rand_idx = np.random.permutation(pcd.shape[0])
        pcd = pcd[rand_idx[: self.point_num]]
        return pcd

    def fetch_mesh(self, objpath: str) -> Tuple[torch.Tensor, torch.LongTensor]:
        meshpath = osp.join(self.rootpath + 'Remesh', objpath + '.off')
        vertices, faces, _ = import_mesh(meshpath)
        return vertices, faces


class TextDataset():
    def __init__(self,
                 path: str = osp.join('assets', 'text')):
        data = json.load(open(osp.join(path, 'embedding.json')))
        self.task = {}
        self.base = {}
        self.cata = data.keys()
        for cata, dict in data.items():
            base_emb = dict['base']
            task_emb = {'embedding': [], 'sentence': []}
            for task, emb in dict.items():
                if task == 'base': continue
                task_emb['embedding'].extend(emb['embedding'])
                task_emb['sentence'].extend(emb['sentence'])
            self.task[cata] = task_emb
            self.base[cata] = base_emb

    def fetch_text_emb(self, cata: str, task_label: bool):
        assert cata in self.cata
        mode = 'task' if task_label else 'base'
        if mode == 'base':
            sentense_num = len(self.base[cata]['embedding'])
            idx = np.random.randint(sentense_num)
            return np.asarray(self.base[cata]['embedding'][idx])
        else:
            sentense_num = len(self.task[cata]['embedding'])
            idx = np.random.randint(sentense_num)
            return np.asarray(self.task[cata]['embedding'][idx])

class GraspDataset(Dataset):

    def __init__(self,
                 path: str = osp.join('assets', 'grasp'),
                 subdir: str = None,
                 object_dataset: ObjectDataset = None,
                 text_dataset: TextDataset = None,
                 require_obj_path: bool = False,
                 data_augmentation: bool = False):
        if subdir is None:
            dirs = os.listdir(path)
            subdirs = [dir for dir in dirs if osp.isdir(osp.join(path, dir))]
        else:
            subdirs = [subdir]

        pcd = []
        pcd_idx = []
        poses = []
        task_label = []
        cata_label = []
        joints = []
        pcd_paths = []
        for cata_idx, dir in enumerate(subdirs):
            subpath = osp.join(path, dir)
            obj_paths = os.listdir(subpath)
            for obj_path in obj_paths:
                obj_pose_file = osp.join(subpath, obj_path, 'success_pose.json')
                obj_joint_file = osp.join(subpath, obj_path, 'success_grab_joint_state.json')
                obj_task_file = osp.join(subpath, obj_path, 'success_pressed.json')
                if not osp.exists(obj_pose_file):
                    continue
                with open(obj_pose_file) as f:
                    pose = np.asarray(json.load(f))
                with open(obj_joint_file) as f:
                    joint = np.asarray(json.load(f))
                with open(obj_task_file) as f:
                    success = np.asarray(json.load(f))
                if len(success) == 0: continue
                if len(pose) > 0:
                    joint = normalized_joint_state(joint)
                    if object_dataset is not None:
                        pcd_path = osp.join(dir, obj_path)
                        pcd_paths.append(pcd_path)
                        pcd.append(object_dataset.fetch_pointcloud(pcd_path))
                    poses.extend(pose)
                    joints.extend(joint)
                    task_label.extend(success)
                    cata_label.extend(len(pose) * [cata_idx])
                    pcd_idx.extend(len(pose) * [len(pcd) - 1])
        self.object_dataset = object_dataset
        self.text_dataset = text_dataset
        self.pcd = np.asarray(pcd)
        self.pose = np.asarray(poses)
        self.joint = np.asarray(joints)
        self.pcd_paths = pcd_paths
        self.task_label = np.asarray(task_label)
        self.cata_label = np.asarray(cata_label)
        self.pcd_idx = np.asarray(pcd_idx, dtype = int)
        self.cata = subdirs
        self.require_obj_path = require_obj_path
        self.data_augmentation = data_augmentation

    def __len__(self):
        return self.pose.shape[0]

    def __getitem__(self, idx):
        ret = [self.pose[idx], self.joint[idx]]
        if self.object_dataset is not None:
            if self.require_obj_path:
                ret.append(self.get_obj_path(idx))
            else:
                pcd = self.pcd[self.pcd_idx[idx]]
                pcd[:, 0] = -pcd[:, 0]
                ret.append(pcd)
        if self.text_dataset is not None:
            ret.append(self.text_dataset.fetch_text_emb(self.cata[self.cata_label[idx]], self.task_label[idx]))
        return tuple(ret)

    def get_obj_path(self, idx):
        return self.pcd_paths[self.pcd_idx[idx]]

def analysis_dataset(datasetpath: str = 'TOG'):
    device = 'cuda:0'
    gripper = GripperModel('shadow', use_complete_points=True)
    object_dataset = ObjectDataset(osp.join('assets', 'object', datasetpath))
    grasp_dataset = GraspDataset(osp.join('assets', 'grasp', datasetpath), object_dataset = object_dataset, require_obj_path=True)
    dataloader = DataLoader(grasp_dataset, batch_size = 1, shuffle = False)
    q1_list = []
    pen_list = []
    for i, (pose, joint, obj_path) in enumerate(tqdm(dataloader)):
        pose = pose.to(device).to(torch.float32)
        joint = joint.to(device).to(torch.float32)
        obj_path = obj_path[0]
        vertices, faces = object_dataset.fetch_mesh(obj_path)
        joint = recover_joint_state(joint) * np.pi / 180.0
        q1 = cal_q1(gripper, pose, joint, vertices, faces)
        pen = cal_pen(gripper, pose, joint, vertices, faces)
        q1_list.extend(q1.tolist())
        pen_list.extend(pen.detach().cpu().numpy().tolist())
    q1 = np.asarray(q1_list)
    pen = np.asarray(pen_list)
    print(f"Q1 Mean: {np.mean(q1)}, penetration mean: {np.mean(pen)}")

if __name__ == '__main__':
    object_dataset = ObjectDataset()
    text_dataset = TextDataset()
    dataset = GraspDataset(osp.join('assets', 'grasp', 'TOG'), subdir="drink", object_dataset=object_dataset, text_dataset=text_dataset)
    gripper = GripperModel('shadow', use_complete_points=True)
    dataloader = DataLoader(dataset, batch_size = 4)
    for i, (pose, joint, pcd, text_emb) in enumerate(tqdm(dataloader)):
        joint = recover_joint_state(joint)
        points, _, _ = gripper.compute_pcd(pose.to(torch.float), joint.to(torch.float) / 180.0 * np.pi)
        for j in range(points.shape[0]):
            gripper_pcd = points[j]
            visualize_point_cloud(pcd[j].detach().cpu().numpy(), gripper_pcd.detach().cpu().numpy())





