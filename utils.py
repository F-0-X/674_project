import os
import torch
import ocnn
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_modelnet40(train=True, suffix='points'):
    root = 'dataset/ModelNet40.points'
    points, labels = [], []
    folders = sorted(os.listdir(root))
    assert len(folders) == 40
    for idx, folder in enumerate(folders):
        subfolder = 'train' if train else 'test'
        current_folder = os.path.join(root, folder, subfolder)
        filenames = sorted(os.listdir(current_folder))
        for filename in filenames:
            if filename.endswith(suffix):
                filename_abs = os.path.join(current_folder, filename)
                points.append(np.fromfile(filename_abs, dtype=np.uint8))
                labels.append(idx)
    return points, labels, folders


class ourDataset:

    def __init__(self, points, labels, phase="train", args=None):
        self.phase = phase
        self.points = points
        self.labels = labels
        self.number_samples = len(points)

        if self.phase == "train":
            self.bs = args.train_batch

    def get_batch(self, idx):
        start_idx = idx * self.bs
        end_idx = min(start_idx + self.bs, self.number_samples)  # exclusive
        if self.phase == 'val':
            curr_points = self.points[start_idx:end_idx, :]
            curr_labels = self.labels[start_idx:end_idx]
        if self.phase == "train":
            random_index = np.random.choice(len(self.points), self.bs)
            curr_points = self.points[random_index]
            curr_labels = self.labels[random_index]
        flags = vars(self.args)
        transform = ocnn.TransformCompose(flags)
        batch = []
        # for each curr_points
        for index, point in enumerate(curr_points):
            # 1. to tensor
            t_point = point.to(device)
            # 2. call transform
            octree = transform(t_point)
            # 3. form a tuple with corresponding label and add to a new list
            batch.append((octree, curr_labels[index]))

            # ocnn.collate_octrees with the new list(batch)
            # the result from it should be super_octree and curr_label
            # return both
            return ocnn.collate_octrees(batch)



if __name__ == '__main__':
    load_modelnet40()
