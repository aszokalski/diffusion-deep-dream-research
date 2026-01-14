from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.length = end - start

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError("Index out of bounds")

        number = self.start + index
        return number
