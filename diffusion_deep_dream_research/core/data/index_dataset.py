from typing import List, Optional, Union

from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, start_or_indices: Union[int, List[int]], end: Optional[int] = None):
        """
        Args:
            start_or_indices: Either a list of integers OR the start of a range.
            end: The end of the range (only required if start_or_indices is an int).
        """
        if isinstance(start_or_indices, list):
            self.numbers = start_or_indices
        else:
            if end is None:
                raise ValueError("If providing a start integer, 'end' must also be provided.")
            self.numbers = range(start_or_indices, end)

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, index):
        return self.numbers[index]
