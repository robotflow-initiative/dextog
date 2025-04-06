import os
import json
import numpy as np
import os.path as osp

from typing import Dict, Tuple, List

def get_data_dict(task: str, root_dir: str = osp.join("assets", "grasp", "TOGSample")) -> Dict[str, Dict[str, np.ndarray]]:
    shared_data = {}
    data_dir = osp.join(root_dir, task)
    obj_ids = os.listdir(data_dir)
    for obj_id in obj_ids:
        id_data_dir = osp.join(data_dir, obj_id)
        pose = np.load(osp.join(id_data_dir, "pose.npy"), allow_pickle=True)
        joint_state = np.load(
            osp.join(id_data_dir, "joint_state.npy"), allow_pickle=True
        )
        with open(
                osp.join(id_data_dir, "success_label.json"), "r", encoding="utf-8"
        ) as f:
            success_label = json.load(f)
        pose = pose[success_label]
        joint_state = joint_state[success_label]
        pose_ = np.sum(pose, axis=1)
        pose_ = np.sum(pose_, axis=1)
        joint_state_ = np.sum(joint_state, axis=1)

        is_nan = np.isnan(pose_ + joint_state_)
        pose = pose[~is_nan]
        joint_state = joint_state[~is_nan]
        if len(pose) != 0 and len(pose) == len(joint_state):
            shared_data[obj_id] = {"pose": pose, "joint_state": joint_state * 180.0 / np.pi}
    return shared_data

class DataIndexLoader(object):

    def __init__(self, data_dict: Dict[str, Dict[str, np.ndarray]]):
        total_len = 0
        data_len = {}
        keys = []
        for key, value in data_dict.items():
            if value["pose"].shape[0] == 0: continue
            keys.append(key)
            data_len[key] = value["pose"].shape[0]
            total_len += value["pose"].shape[0]

        self.keys = keys
        self.data = data_dict
        self.data_len = data_len
        self.total_len = total_len

        self.init()

    def init(self):
        self.cate_idx = 0
        self.idx = 0

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        raise NotImplementedError

    def _update_idx(self, cate_p: int, p: int) -> Tuple[int, int]:
        if p == self.data_len[self.keys[cate_p]]:
            p = 0
            cate_p += 1
        if cate_p == len(self.keys):
            cate_p = 0
        return cate_p, p

    def get_batch(self, batch_size: int = 32) -> Tuple[List[str], List[int]]:
        cate_p = self.cate_idx
        p = self.idx

        data2load = batch_size

        cate_list = []
        idx_list = []
        while self.data_len[self.keys[cate_p]] - p < data2load:
            data_len = self.data_len[self.keys[cate_p]]
            cate_list.extend([self.keys[cate_p] for _ in range(data_len - p)])
            idx_list.extend(range(p, data_len))
            p = data_len
            data2load -= data_len - p
            cate_p, p = self._update_idx(cate_p, p)
        if data2load > 0:
            end_idx = p + data2load
            cate_list.extend([self.keys[cate_p] for _ in range(p, end_idx)])
            idx_list.extend(range(p, end_idx))
            p = end_idx
        assert(len(cate_list) == len(idx_list))
        self.cate_idx = cate_p
        self.idx = p
        return cate_list, idx_list

    def get_success_dict(self) -> Dict:
        success_dict = {key: np.zeros(self.data_len[key], dtype=bool) for key in self.keys}
        return success_dict

