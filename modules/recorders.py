import os
import csv
import pandas as pd

class HistoryRecorder:
    def __init__(self, *, save_base: str, fieldnames: list[str]) -> None:
        self.save_path = os.path.join(save_base, "history.csv")
        self.fieldnames = fieldnames
        self.record = []

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def add_record(self, rowdict: dict) -> None:
        self.record.append(rowdict)
        with open(self.save_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(rowdict)
    
    # def get_record_df(self) -> pd.DataFrame:
    #     return pd.concat(self.record, )