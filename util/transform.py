import numpy as np

import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, label):
        for t in self.transforms:
            data, label = t(data, label)
        return data, label


class ToTensor(object):
    def __call__(self, data, label):
        data = torch.from_numpy(data)
        if not isinstance(data, torch.FloatTensor):
            data = data.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return data, label


class RandomRotate(object):
    def __init__(self, rotate_angle=None, along_z=False):
        self.rotate_angle = rotate_angle
        self.along_z = along_z

    def __call__(self, data, label):
        if self.rotate_angle is None:
            rotate_angle = np.random.uniform() * 2 * np.pi
        else:
            rotate_angle = self.rotate_angle
        cosval, sinval = np.cos(rotate_angle), np.sin(rotate_angle)
        if self.along_z:
            rotation_matrix = np.array([[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]])
        else:
            rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        data[:, 0:3] = np.dot(data[:, 0:3], rotation_matrix)
        if data.shape[1] > 3:  # use normal
            data[:, 3:6] = np.dot(data[:, 3:6], rotation_matrix)
        return data, label


class RandomRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data, label):
        angles = np.clip(self.angle_sigma*np.random.randn(3), -self.angle_clip, self.angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        data[:, 0:3] = np.dot(data[:, 0:3], R)
        if data.shape[1] > 3:  # use normal
            data[:, 3:6] = np.dot(data[:, 3:6], R)
        return data, label


class RandomScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.25):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, data, label):
        scale = np.random.uniform(self.scale_low, self.scale_high)
        data[:, 0:3] *= scale
        return data, label


class RandomShift(object):
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data, label):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data[:, 0:3] += shift
        return data, label


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data, label):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(data.shape[0], 3), -1 * self.clip, self.clip)
        data[:, 0:3] += jitter
        return data, label
