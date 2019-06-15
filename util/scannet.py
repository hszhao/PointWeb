import pickle
import os
import numpy as np

from torch.utils.data import Dataset


class ScanNet(Dataset):
    def __init__(self, split='train', data_root='scannet', num_point=8192, classes=20, block_size=1.5, sample_rate=1.0, transform=None):
        self.split = split
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        data_file = os.path.join(data_root, 'scannet_{}.pickle'.format(split))
        file_pickle = open(data_file, 'rb')
        xyz_all = pickle.load(file_pickle, encoding='latin1')
        label_all = pickle.load(file_pickle, encoding='latin1')
        file_pickle.close()

        self.label_all = []  # for change 0-20 to 0-19 + 255
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        label_weight = np.zeros(classes+1)
        for index in range(len(xyz_all)):
            xyz, label = xyz_all[index], label_all[index]  # xyzrgb, N*6; l, N
            coord_min, coord_max = np.amin(xyz, axis=0)[:3], np.amax(xyz, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(label.size)
            tmp, _ = np.histogram(label, range(classes + 2))
            label_weight += tmp
            label_new = label - 1
            label_new[label == 0] = 255
            self.label_all.append(label_new.astype(np.uint8))
        label_weight = label_weight[1:].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = 1 / np.log(1.2 + label_weight)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(xyz_all)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        self.xyz_all = xyz_all
        self.label_weight = label_weight
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.xyz_all[room_idx]  # N * 3
        labels = self.label_all[room_idx]  # N
        N_points = points.shape[0]

        for i in range(10):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_min[2], block_max[2] = self.room_coord_min[room_idx][2], self.room_coord_max[room_idx][2]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size == 0:
                continue
            vidx = np.ceil((points[point_idxs, :] - block_min) / (block_max - block_min) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
            if ((labels[point_idxs] != 255).sum() / point_idxs.size >= 0.7) and (vidx.size/31.0/31.0/62.0 >= 0.02):
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 3
        current_points = np.zeros((self.num_point, 6))  # num_point * 6
        current_points[:, 3] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 4] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 5] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:3] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    data_root = '/mnt/sda1/hszhao/dataset/scannet'
    point_data = ScanNet(split='train', data_root=data_root, num_point=8192, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(2):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()
