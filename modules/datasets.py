import torch
from torch.utils.data import Dataset

class LogDataset(Dataset):
    def __init__(self, *, logs: list[str], labels: list[int]):
        super().__init__()
        assert len(logs) == len(labels)
        self.logs = logs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "log": self.logs[index],
            "label": torch.tensor(self.labels[index], dtype=torch.float)
        }
    
class LogWindowDataset(Dataset):
    def __init__(self, *, log_wins: list[list[str]], labels: list[str]):
        super().__init__()
        assert len(log_wins) == len(labels)
        self.log_wins = log_wins
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def collate_fu(self, batch):
        return {
            "log_win": [item["log_win"] for item in batch],
            "label": [item["label"] for item in batch],
        }

    def __getitem__(self, index):
        return {
            "log_win": self.log_wins[index],
            "label": self.labels[index],
        }
