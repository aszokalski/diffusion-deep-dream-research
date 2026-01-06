from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.length = end - start

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of bounds")

        number = self.start + idx

        return number