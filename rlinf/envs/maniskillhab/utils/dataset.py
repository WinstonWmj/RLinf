from abc import abstractmethod

from torch.utils.data import DataLoader, Dataset


class ClosableDataset(Dataset):
    @abstractmethod
    def close(self): ...


class ClosableDataLoader(DataLoader):
    def close(self):
        self.dataset.close()
