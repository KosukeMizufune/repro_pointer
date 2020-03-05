from pathlib import Path

from torch.utils.data import Dataset
import numpy as np
import h5py


class ModelNet40(Dataset):
    def __init__(self, root='data/', train=True, n_points=2048,
                 transforms=None, **kwargs):
        super().__init__()
        self.split = "train" if train else "test"
        self.root = Path(root)
        self.data_root = self.root / 'modelnet40_ply_hdf5_2048'
        self.points, self.labels = self._parse_files()
        self.n_points = n_points
        self.transforms = transforms
        self.n_class = 40

    def _get_data_files(self, list_filename):
        with open(list_filename) as f:
            return [line.rstrip()[5:] for line in f]

    def _load_data_file(self, filename):
        name = self.root / filename
        f = h5py.File(name)
        data = f["data"][:]
        label = f["label"][:]
        return data, label

    def _parse_files(self):
        files = self._get_data_files(self.data_root / f"{self.split}_files.txt")
        point_list, label_list = [], []
        for f in files:
            points, labels = self._load_data_file(f)
            point_list.append(points)
            label_list.append(labels)
        return np.concatenate(point_list, 0), np.concatenate(label_list, 0)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.n_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        if self.transforms:
            current_points = self.transforms(current_points)
        label = self.labels[idx, 0].astype(np.int)
        return current_points, label

    def __len__(self):
        return len(self.points)
