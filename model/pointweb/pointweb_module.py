from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops
from util import pt_util


class _AFAModule(nn.Module):
    def __init__(self, mlp, use_softmax=False):
        r"""
        :param mlp: mlp for learning weight
               mode: transformation or aggregation
        """
        super().__init__()
        self.mlp = mlp
        self.use_softmax = use_softmax

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N, M) or (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
            transformation: (B, C, N, M) or (B, C, N)
            aggregation: (B, C, N) or (B, C)
        """
        B, C, N, M = feature.size()
        feature = feature.transpose(1, 2).contiguous().view(B * N, C, M, 1).repeat(1, 1, 1, M)  # (BN, C, M, M)
        feature = feature - feature.transpose(2, 3).contiguous() + torch.mul(feature, torch.eye(M).view(1, 1, M, M).cuda())  # (BN, C, M, M)
        weight = self.mlp(feature)
        if self.use_softmax:
            weight = F.softmax(weight, -1)
        feature = (feature * weight).sum(-1).view(B, N, C, M).transpose(1, 2).contiguous()  # (B, C, N, M)
        return feature


class _PointWebSAModuleBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.npoint = None
        self.grouper = None
        self.mlp = None
        self.afa = None

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        new_xyz = pointops.gathering(
            xyz_trans,
            pointops.furthestsampling(xyz, self.npoint)
        ).transpose(1, 2).contiguous() if self.npoint is not None else None
        new_features = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        if new_features.shape[2] != 1:  # for npoint is none
            new_features = new_features + self.afa(new_features)  # (B, C, npoint, nsample)
        new_features = self.mlp(new_features)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(-1)  # (B, mlp[-1], npoint)
        new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointWebSAModule(_PointWebSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping
    Parameters
    ----------
    npoint : int
        Number of features
    nsample : int32
        Number of sample
    mlps : list of int32
        Spec of the MLP before the global max_pool
    mlps2: list of list of int32
        Spec of the MLP for AFA
    bn : bool
        Use batchnorm
    """
    def __init__(self, *, npoint: int = None, nsample: int = None, mlp: List[int] = None, mlp2: List[int] = None, bn: bool = True, use_xyz: bool = True, use_bn = True):
        super().__init__()
        self.npoint = npoint
        self.grouper = pointops.QueryAndGroup(nsample=nsample, use_xyz=use_xyz) if npoint is not None else pointops.GroupAll(use_xyz)
        if use_xyz:
            mlp[0] += 3
        if npoint is not None:
            mlp_tmp = pt_util.SharedMLP([mlp[0]] + mlp2, bn=use_bn)
            mlp_tmp.add_module('weight', (pt_util.SharedMLP([mlp2[-1], mlp[0]], bn=False, activation=None)))
            self.afa = _AFAModule(mlp=mlp_tmp)
        self.mlp = pt_util.SharedMLP(mlp, bn=bn)


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    c = 6
    xyz = torch.randn(2, 8, 3, requires_grad=True).cuda()
    xyz_feats = torch.randn(2, 8, c, requires_grad=True).cuda()

    test_module = PointWebSAModule(npoint=2, nsample=6, mlp=[c, 32, 32], mlp2=[16, 16], use_bn=True)
    test_module.cuda()
    xyz_feats = xyz_feats.transpose(1, 2).contiguous()
    print(test_module)
    print(test_module(xyz, xyz_feats))

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
