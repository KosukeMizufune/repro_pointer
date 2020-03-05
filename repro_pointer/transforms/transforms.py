import numpy as np
import torch


def rodrigues_rotation(angle, axis_vec):
    u = axis_vec / np.linalg.norm(axis_vec)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    rotation = cosval * np.eye(3) + sinval * cross_prod_mat \
        + (1.0 - cosval) * np.outer(u, u)
    return rotation


class RandomRotate:
    def __call__(self, points):
        axis_vec = np.random.rand(3)
        rotate_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = rodrigues_rotation(rotate_angle, axis_vec)
        return np.matmul(points, rotation_matrix.T)


class RandomScale:
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class RandomJitter:
    def __init__(self, std=0.01, clip=0.05):
        self.std = std
        self.clip = clip

    def __call__(self, points):
        jittered = np.clip(np.random.normal(0, self.std, points.shape),
                           -self.clip, self.clip)
        points[:, 0:3] += jittered
        return points


class RandomTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range,
                                        self.translate_range)
        points[:, 0:3] += translation
        return points


class ToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).permute(1, 0).to(dtype=torch.float32)
