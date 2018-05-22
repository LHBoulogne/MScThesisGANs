from torch.utils.data import Dataset
from torchvision import datasets

class Subset(Dataset) :
    def __init__(self, dset, idcs):
        self.dset = dset
        self.idcs = idcs

    def __len__(self):
        return len(self.idcs)

    # Uses the full dataset to get a random labelbatch. This should be no problem since the subset is chosen randomly
    def get_random_labelbatch(self, batch_size):
        return self.dset.get_random_labelbatch(batch_size)

    def __getitem__(self, idx):
        idx2 = self.idcs[idx]
        return self.dset[idx2]