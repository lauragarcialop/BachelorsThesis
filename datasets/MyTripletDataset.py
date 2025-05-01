import torch
from numpy.random import randint
from torch.utils.data import Dataset
from functions.DS_Functions import read_dataset


class TripletDataset(Dataset):
    def __init__(self, filename, p=0.5, n=0.5):
        """
        Args:
            data (array-like): simulation
            labels (array-like): parameters that generated it
        """
        self.labels, self.anchor = read_dataset(filename)

        self.positive, self.negative = torch.zeros_like(self.anchor), torch.zeros_like(self.anchor)

        L = self.labels.shape[0]
        P, N = int(p * (L-1)), int(n * (L-1))
        for i, theta in enumerate(self.labels):
            # find distances between parameters and sort the resulting vector
            d = torch.norm(self.labels-theta.unsqueeze(0), p=2, dim=1)
            _, indices = torch.sort(d)

            # randomly choose a positive sample and a negative sample and add them as the 
            kp, kn = randint(1, P+1), randint(L-N, L)
            self.positive[i], self.negative[i] = self.anchor[indices[kp]], self.anchor[indices[kn]]


    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        sample = (self.anchor[idx], self.positive[idx], self.negative[idx])
        label = self.labels[idx]

        return sample, label