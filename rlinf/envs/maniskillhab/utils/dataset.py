# Copyright 2025 ManiSkill-HAB Authors.
#
# wei mingjie copy from https://github.com/arth-shukla/mshab/tree/main and make some revise

from abc import abstractmethod

from torch.utils.data import DataLoader, Dataset


class ClosableDataset(Dataset):
    @abstractmethod
    def close(self): ...


class ClosableDataLoader(DataLoader):
    def close(self):
        self.dataset.close()
