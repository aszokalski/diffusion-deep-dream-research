import json
from pathlib import Path

from torch.utils.data import Dataset


class UniquePromptDataset(Dataset):
    def __init__(self, jsonl_file: Path):
        self.prompts = self._load_unique_prompts(jsonl_file)

    def _load_unique_prompts(self, jsonl_file: Path):
        unique_prompts = set()

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'text' in data:
                                unique_prompts.add(data['text'])
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON line: {line[:50]}...")
                            continue

            return sorted(list(unique_prompts))

        except FileNotFoundError:
            print(f"Error: The file {jsonl_file} was not found.")
            return []

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]