from torch.utils.data import Dataset
from functions.DS_Functions import read_dataset


class Dataset(Dataset):
    def __init__(self, filename):
        """
        Args:
            data (array-like): simulation
            labels (array-like): parameters that generated it
        """
        self.labels, self.data = read_dataset(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return sample, label