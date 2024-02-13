import torch

# def skew(w):
#     """Build a skew matrix ("cross product matrix") for vector w.
#     Args:
#         w: (3,) A 3-vector

#     Returns:
#         W: (3, 3) A skew matrix such that W @ v == w x v
#     """
#     w = w.reshape(3)
#     return torch.tensor([[0.0, -w[2], w[1]], 
#                          [w[2], 0.0, -w[0]], 
#                          [-w[1], w[0], 0.0]], device=w.device)

def skew(w):
    """Build a skew matrix ("cross product matrix") for batch of vector w."""
    w = w.reshape(w.shape[0], 3)
    zeros = torch.zeros(w.shape[0], device=w.device)
    return torch.stack([
        torch.stack([zeros, -w[:,2], w[:,1]], dim=1),
        torch.stack([w[:,2], zeros, -w[:,0]], dim=1),
        torch.stack([-w[:,1], w[:,0], zeros], dim=1)
    ], dim=1)


# def rp_to_se3(R, p):
#     """Rotation and translation to homogeneous transform.
#     Args:
#         R: (3, 3) An orthonormal rotation matrix.
#         p: (3,) A 3-vector representing an offset.

#     Returns:
#         X: (4, 4) The homogeneous transformation matrix described by rotating by R
#             and translating by p.
#     """
#     p = p.reshape(3, 1)
#     bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=p.device)
#     return torch.cat([torch.cat([R, p], dim=1), bottom_row], dim=0)

def rp_to_se3(R, p):
    """Batch of rotation and translation to homogeneous transform."""
    p = p.view(p.shape[0], 3, 1)
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=p.device).repeat(p.shape[0], 1).unsqueeze(1)
    return torch.cat([torch.cat([R, p], dim=2), bottom_row], dim=1)


# def exp_so3(w, theta):
#     """Exponential map from Lie algebra so3 to Lie group SO3.
#     Args:
#         w: (3,) An axis of rotation.
#         theta: An angle of rotation.

#     Returns:
#         R: (3, 3) An orthonormal rotation matrix representing a rotation of
#             magnitude theta about axis w.
#     """
#     W = skew(w)
#     return torch.eye(3, device=w.device) + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * torch.mm(W, W)

def exp_so3(w, theta):
    """Exponential map from batch of Lie algebra so3 to Lie group SO3."""
    W = skew(w)
    I = torch.eye(3, device=w.device).unsqueeze(0).repeat(w.shape[0], 1, 1)
    return I + torch.sin(theta).unsqueeze(-1).unsqueeze(-1) * W + (1.0 - torch.cos(theta).unsqueeze(-1).unsqueeze(-1)) * torch.bmm(W, W)


# def exp_se3(S, theta):
#     """Exponential map from Lie algebra se3 to Lie group SE3.
#     Args:
#         S: (6,) A screw axis of motion.
#         theta: Magnitude of motion.

#     Returns:
#         a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
#             motion of magnitude theta about S for one second.
#     """
#     w, v = torch.split(S, 2)
#     W = skew(w)
#     R = exp_so3(w, theta)
#     p = (theta * torch.eye(3, device=S.device) + (1.0 - torch.cos(theta)) * W +
#          (theta - torch.sin(theta)) * torch.mm(W, W)) @ v
#     return rp_to_se3(R, p)


def exp_se3(S, theta):
    """Exponential map from batch of Lie algebra se3 to Lie group SE3."""
    w, v = torch.split(S, 3, dim=1)
    W = skew(w)
    R = exp_so3(w, theta)
    I = torch.eye(3, device=S.device).unsqueeze(0).repeat(S.shape[0], 1, 1)
    p = torch.bmm(theta.unsqueeze(-1).unsqueeze(-1) * I + (1.0 - torch.cos(theta).unsqueeze(-1).unsqueeze(-1)) * W + (theta.unsqueeze(-1).unsqueeze(-1) - torch.sin(theta).unsqueeze(-1).unsqueeze(-1)) * torch.bmm(W, W), v.unsqueeze(-1)).squeeze(-1)
    return rp_to_se3(R, p)


def to_homogenous(v):
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)

def from_homogenous(v):
    return v[..., :3] / v[..., -1:]


# def to_homogenous(v):
#     return torch.cat([v, torch.ones(v.shape[0], 1, device=v.device)], dim=1)

# def from_homogenous(v):
#     return v[..., :3] / v[..., -1:].unsqueeze(-1)
