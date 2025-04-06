import torch
import kaolin
import numpy as np

from typing import Tuple
from torch.optim import SGD
from scipy.spatial import ConvexHull

from ..data.utils import GripperModel

def cal_q1(gripper: GripperModel,
               Tbase: torch.Tensor,
               joints: torch.Tensor,
               vertices: torch.Tensor,
               faces: torch.LongTensor,
               dist_threshold: float = 0.05,
               mu: float = 1.0,
               lambda_torque: float = 10,
               sparse: bool = True) -> np.ndarray:
    """
    :param gripper:
    :param Tbase: wrist transformation matrix, (batch_size, 4, 4)
    :param joints: joint parameter, (batch_size, gripper_dof)
    :param vertices: object mesh vertices (num_points, 3)
    :param faces: object mesh faces (num_faces, num_vertices)
    :param dist_threshold: distance threshold for q1 calculation, the point with distance below this threshold will be taken into account
    :param mu: friction coefficient for q1 calculation
    :param lambda_torque: ratio of torque and force for q1 calculation
    :param sparse: if true, use one contact point per link
    :return: q1s: calculated q1, (batch_size)
    """
    device = joints.device
    faces = faces.to(device)
    vertices = vertices.to(device)

    hand_pcd, _, _ = gripper.compute_pcd(Tbase, joints, )
    batch_size, num_points, _ = hand_pcd.shape

    vertices = vertices.repeat((batch_size, 1, 1))
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces) # (batch_size, num_faces, 3, 3)

    face_normals = kaolin.ops.mesh.face_normals(face_vertices) # (batch_size, num_faces, 3)
    face_normals /= torch.linalg.norm(face_normals, dim=2, keepdim=True)

    # dist: (batch_size, num_points); face_indices: (batch_size, num_points)
    dist, face_indices, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(hand_pcd, face_vertices)
    dist = torch.sqrt(dist + 1e-8)
    dist_sign = kaolin.ops.mesh.check_sign(vertices, faces, hand_pcd)
    dist = dist * (dist_sign * torch.ones_like(dist) - torch.logical_not(dist_sign) * torch.ones_like(dist)) # (batch_size, num_points)

    idx_tmp = torch.arange(0, batch_size, dtype=torch.long, device=device).reshape(batch_size, 1).repeat(1, hand_pcd.shape[1])
    normals = face_normals[idx_tmp, face_indices]   # (batch_size, num_points, 3)

    if sparse:
        point_per_link = gripper.sample_point_number
        link_num = num_points // point_per_link
        link_dist = torch.zeros((batch_size, link_num)).to(device)
        link_point = torch.zeros((batch_size, link_num, 3)).to(device)
        link_normal = torch.zeros((batch_size, link_num, 3)).to(device)
        idx_tmp = torch.arange(0, batch_size, dtype=torch.long, device=device).reshape(batch_size, 1)
        for link_idx in range(link_num):
            begin_point = link_idx * point_per_link
            end_point = begin_point + point_per_link
            local_dist = dist[:, begin_point: end_point]
            local_normal = normals[:, begin_point: end_point]
            local_point = hand_pcd[:, begin_point: end_point]
            min_idx = torch.argmin(local_dist, dim = 1, keepdim=True)
            link_dist[:, link_idx] += local_dist[idx_tmp, min_idx].reshape(-1)
            link_point[:, link_idx] += local_point[idx_tmp, min_idx].reshape(-1, 3)
            link_normal[:, link_idx] += local_normal[idx_tmp, min_idx].reshape(-1, 3)
        dist = link_dist
        normals = link_normal
        hand_pcd = link_point
        num_points = hand_pcd.shape[1]

    object_points = find_closest_point(hand_pcd, vertices)

    # Sample the force cone
    u1 = torch.zeros_like(normals)
    u2 = torch.zeros_like(normals)
    u1[:, :, 0] -= normals[:, :, 1]
    u1[:, :, 1] += normals[:, :, 0]
    u2[:, :, 0] += torch.ones_like(dist)

    u = torch.where(torch.linalg.norm(u1, dim=2, keepdim=True) > 1e-8, u1, u2)
    u = u / torch.linalg.norm(u, dim=2, keepdim=True)

    normals = normals.reshape(batch_size, num_points, 1, 3)
    u = u.reshape(batch_size, num_points, 1, 3)
    v = torch.linalg.cross(normals, u).reshape(batch_size, num_points, 1, 3)

    theta = torch.linspace(0, 1.75 * np.pi, 8).reshape(1, 1, 8, 1).to(device)
    forces = (normals + mu * (torch.cos(theta) * u + torch.sin(theta) * v))

    contact_mask = dist < dist_threshold
    object_points = object_points.reshape(batch_size, num_points, 1, 3).repeat(1, 1, 8, 1)

    # Filter the contact points and compute the wrench space
    contact_points = torch.zeros_like(object_points)
    contact_forces = torch.zeros_like(forces)
    contact_points[contact_mask] = object_points[contact_mask]
    contact_forces[contact_mask] = forces[contact_mask]
    contact_points = contact_points.reshape(batch_size, -1, 3)
    contact_forces = contact_forces.reshape(batch_size, -1, 3)
    contact_torques = lambda_torque * torch.linalg.cross(contact_forces, contact_points)
    contact_wrenches = torch.cat((contact_forces, contact_torques), dim = 2).detach().cpu().numpy()

    # Using the wrench space to compute the q1
    q1s = []
    origin = np.asarray([[0, 0, 0, 0, 0, 0]])
    for i in range(batch_size):
        wrenches = contact_wrenches[i]
        wrenches = np.unique(wrenches, axis = 0)
        if not sparse:
            wrenches = wrenches[np.random.permutation(len(wrenches))[: 1023]]
        wrenches = np.concatenate((wrenches, origin), axis = 0)
        try:
            wrench_space = ConvexHull(wrenches)
            q1 = np.asarray(1.0)
            for equation in wrench_space.equations:
                q1 = np.minimum(q1, np.abs(equation[6]) / np.linalg.norm(equation[:6]))
            q1s.append(q1)
        except:
            q1s.append(np.asarray(0.0))
    q1s = np.asarray(q1s)
    return q1s

def find_closest_point(src_points: torch.Tensor,
                       dest_points: torch.Tensor) -> torch.Tensor:
    """
    :param srcs_point: (B, N1, 3)
    :param dests_point: (B, N2, 3)
    :return: (B, N1, 3)
    """
    B, N1, _ = src_points.shape
    _, N2, _ = dest_points.shape

    d_1 = torch.tile(torch.sum(src_points ** 2, dim=2).unsqueeze(2), (1, 1, N2))
    d_2 = torch.tile(torch.sum(dest_points ** 2, dim=2).unsqueeze(1), (1, N1, 1))
    d_3 = torch.einsum('ijk,ilk->ijl', src_points, dest_points)
    d = d_1 + d_2 - 2 * d_3
    idx = torch.argmin(d, dim=2)

    batch_idx = torch.arange(B).reshape(B, 1).repeat(1, N1)
    return dest_points[batch_idx, idx]

def cal_pen(gripper: GripperModel,
               Tbase: torch.Tensor,
               joints: torch.Tensor,
               vertices: torch.Tensor,
               faces: torch.LongTensor) -> torch.Tensor:
    """
    Calculate the maximum penetration between object and hand
    :param gripper:
    :param Tbase: wrist transformation matrix, (batch_size, 4, 4)
    :param joints: joint parameter, (batch_size, gripper_dof)
    :param vertices: object mesh vertices (num_points, 3)
    :param faces: object mesh faces (num_faces, num_vertices)
    :return: pen: max penetration depth, (batch_size)
    """
    device = joints.device
    faces = faces.to(device)
    vertices = vertices.to(device)
    batch_size = joints.shape[0]

    pen_obj = cal_pen_obj(gripper, Tbase, joints, vertices, faces)
    vertices = vertices.repeat((batch_size, 1, 1))

    pen_hand = cal_pen_hand(gripper, Tbase, joints, vertices)

    return torch.maximum(pen_obj, pen_hand)

def cal_pen_obj(gripper: GripperModel,
               Tbase: torch.Tensor,
               joints: torch.Tensor,
               vertices: torch.Tensor,
               faces: torch.LongTensor) -> torch.Tensor:
    """
    Calculate the penetration depth using the object mesh and hand pcd
    :param gripper:
    :param Tbase: wrist transformation matrix, (batch_size, 4, 4)
    :param joints: joint parameter, (batch_size, gripper_dof)
    :param vertices: object mesh vertices (num_points, 3)
    :param faces: object mesh faces (num_faces, num_vertices)
    :return: pen: max penetration depth, (batch_size, )
    """
    device = joints.device
    faces = faces.to(device)
    vertices = vertices.to(device)

    hand_pcd, _, _ = gripper.compute_pcd(Tbase, joints, )
    batch_size, num_points, _ = hand_pcd.shape

    vertices = vertices.repeat((batch_size, 1, 1))
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)  # (batch_size, num_faces, 3, 3)

    pen = torch.zeros((batch_size,), dtype=torch.float32).to(device)

    # dist: (batch_size, num_points); face_indices: (batch_size, num_points)
    dist, face_indices, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(hand_pcd, face_vertices)
    dist = torch.sqrt(dist + 1e-8)
    dist_sign = kaolin.ops.mesh.check_sign(vertices, faces, hand_pcd)
    dist_tmp = torch.zeros_like(dist)
    dist_tmp[dist_sign] += dist[dist_sign]
    pen = torch.maximum(pen, torch.max(dist_tmp, dim = 1)[0])

    return pen

def cal_pen_hand(gripper: GripperModel,
                 Tbase: torch.Tensor,
                 joints: torch.Tensor,
                 obj_pcd: torch.Tensor) -> torch.Tensor:
    """
    Calculate the penetration depth using hand mesh and object pcd
    :param gripper:
    :param Tbase: wrist transformation matrix, (batch_size, 4, 4)
    :param joints: joint parameter, (batch_size, gripper_dof)
    :param obj_pcd: object point cloud, (num_point, 3)
    :return: pen: max penetration depth, (batch_size, )
    """
    device = joints.device
    batch_size = Tbase.shape[0]

    vertices_batch, faces_batch = gripper.compute_mesh(Tbase, joints)
    mesh_num = len(vertices_batch)

    pen = torch.zeros((batch_size, ), dtype=torch.float32).to(device)
    for i in range(mesh_num):
        vertices = vertices_batch[i] # (batch_size, num_vertices, 3)
        faces = faces_batch[i]      # (num_faces, 3)
        face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices, faces)
        dist, face_indices, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_pcd, face_vertices)
        dist = torch.sqrt(dist + 1e-8)  # (batch_size, obj_point_num)
        dist_sign = kaolin.ops.mesh.check_sign(vertices, faces, obj_pcd)
        dist_tmp = torch.zeros_like(dist)
        dist_tmp[dist_sign] += dist[dist_sign]
        pen = torch.maximum(pen, torch.max(dist_tmp, dim = 1)[0])
    return pen

def TTA(gripper: GripperModel,
        Tbase: torch.Tensor,
        joints: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.LongTensor,
        iter_num: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param gripper:
    :param Tbase: wrist transformation matrix, (batch_size, 4, 4)
    :param joints: joint parameter, (batch_size, gripper_dof)
    :param vertices: object mesh vertices (num_points, 3)
    :param faces: object mesh faces (num_faces, num_vertices)
    :return: [Tbase_, joints_]: optimized Tbase and joints_
    """
    joints.requires_grad_(True)
    optimizer = SGD([joints], lr = 1e-1, momentum = 0.0)
    for _ in range(iter_num):
        optimizer.zero_grad()
        loss = torch.sum(cal_pen_obj(gripper, Tbase, joints, vertices, faces))
        loss.backward()
        optimizer.step()
    return Tbase, joints

