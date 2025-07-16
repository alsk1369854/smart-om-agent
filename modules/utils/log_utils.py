import pandas as pd
import re
from typing import  Callable, Optional, Callable
from tqdm import tqdm
from .. import types

LOG_REPLACE_PATTERN = "|".join([
    r'True', r'true', r'False', r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?', #  IP
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}', # Special MAC
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}', # MAC
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*', # File Path
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*', # word have number
])
LOG_REPLACE_REGEX = re.compile(LOG_REPLACE_PATTERN)
DOTS_REGEX = re.compile(r'[\.]{3,}')
def log_regx_replase(log: str) -> str:
    log = re.sub(DOTS_REGEX, '.. ', log)   # Replace multiple '.' with '.. '
    log = re.sub(LOG_REPLACE_PATTERN, '<*>', log)
    return log

class LogDataFrameHelper:
    def __init__(self, bast_log_path: str):
        self.bast_log_path = bast_log_path

    # struct
    def get_struct_logs_path(self) -> str:
        return self.bast_log_path + "_struct_logs.csv"
    
    def build_struct_logs(
        self,
        log_path: str,
        log_fromat: str, # e.g. "<label> <timestamp> <date> <node> <time> <node_repeat> <type> <component> <level> <content>",
        start_line: int = 0,
        end_line: Optional[int] = None,
    ) -> pd.DataFrame:
        fieldnames = log_fromat.split(' ')
        fieldnames = [f.strip('<>') for f in fieldnames]
        fieldnames_len = len(fieldnames)
        data = []
        lines_count = -1
        with open(log_path, 'r', encoding='latin-1') as f:
            while True:
                line = f.readline()
                lines_count += 1
                if not line or (lines_count >= end_line):
                    break
                if start_line > lines_count:
                    continue

                line.strip()
                cells = re.split(r'\s+', line)
                struct_line = cells[:fieldnames_len-1] + [" ".join(cells[fieldnames_len-1:]).strip()]
                data.append(struct_line)
        return pd.DataFrame(data, columns=fieldnames)

    def save_struct_logs(self, df: pd.DataFrame):
        df.to_csv(self.get_struct_logs_path(), index=False, chunksize=10000)
    
    def load_struct_logs(self) -> pd.DataFrame:
        return pd.read_csv(self.get_struct_logs_path())
    
    # semantic_logs
    def get_semantic_logs_path(self, dtype: Optional[types.DatasetTypes] = None) -> str:
        suffix = "_semantic_logs.csv"
        suffix = suffix if dtype is None else f"_{dtype}" + suffix
        return self.bast_log_path + suffix
    
    def build_semantic_logs(
        self,
        struct_df: pd.DataFrame,
        feature_columns: list[str],
        log_rex_replase_fn: Callable[[str], str]
    ) -> pd.DataFrame:
        struct_df = struct_df.copy()
        data = []
        for _, row in tqdm(struct_df.iterrows(), total=len(struct_df), desc=f"Building semantic logs"):
            timestamp = row["timestamp"]
            log = ", ".join(row[feature_columns])
            log = log_rex_replase_fn(log)
            label = row["label"]
            data.append([timestamp, log, label])
        return pd.DataFrame(data, columns=["timestamp", "log", "label"])

    def save_semantic_logs(self, df: pd.DataFrame, dtype: Optional[types.DatasetTypes] = None):
        df.to_csv(self.get_semantic_logs_path(dtype), index=False, chunksize=10000)
    
    def load_semantic_logs(self, dtype: Optional[types.DatasetTypes] = None) -> pd.DataFrame:
        return pd.read_csv(self.get_semantic_logs_path(dtype=dtype)).astype({"log": str})
    
    # train_adllm_wins
    def get_train_adllm_wins_path(self, dtype: types.WinDatasetTypes, wtype: types.SlidingWindowTypes) -> str:
        suffix = "_train_adllm_wins.csv"
        suffix = f"_{wtype}" + suffix
        suffix = f"_{dtype}" + suffix
        return self.bast_log_path + suffix

    def build_time_winds(
        self,
        df: pd.DataFrame,
        win_size_secs: int = 60 * 60,  # 每個 window 的時間長度 (秒)
        win_step_secs: int = 30 * 60,  # 每個 window 的步進 (秒)
    ) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.sort_values(by=["timestamp"], ascending=True, inplace=True)
        # 預估總共需要幾個 window
        start_time = df["timestamp"].min()
        end_time = df["timestamp"].max()
        total_seconds = (end_time - start_time).total_seconds()
        total_windows = int(total_seconds // win_step_secs) + 1  
        # 開始分割 window
        current_time = start_time
        win_dfs = []
        for index in tqdm(range(total_windows), desc="Building time windows"):
            # 建立窗口區間
            win_mask = (current_time <= df["timestamp"]) & (df["timestamp"] < (current_time + pd.Timedelta(seconds=win_size_secs)))
            win_df = df[win_mask].copy()
            win_df["win_id"] = index
            win_dfs.append(win_df)
            current_time += pd.Timedelta(seconds=win_step_secs)
        return pd.concat(win_dfs, ignore_index=True)
    
    
    def build_train_adllm_time_wins(
        self,
        semantic_logs_df: pd.DataFrame,
        lem_score_map: dict[str, float],
        win_size_secs: int,
        win_step_secs: int,
    ) -> pd.DataFrame:
        semantic_logs_df = semantic_logs_df.copy()
        semantic_logs_df["binary_label"] = semantic_logs_df["label"].apply(lambda x: 0 if x == "-" else 1)
        semantic_logs_df["lem_score"] = semantic_logs_df["log"].apply(lambda x: lem_score_map[x])
        semantic_logs_df["timestamp"] = pd.to_datetime(semantic_logs_df["timestamp"], unit="s")
        semantic_logs_df.sort_values(by=["timestamp"], ascending=True, inplace=True)
        # 預估總共需要幾個 window
        start_time = semantic_logs_df["timestamp"].min()
        end_time = semantic_logs_df["timestamp"].max()
        total_seconds = (end_time - start_time).total_seconds()
        total_windows = int(total_seconds // win_step_secs) + 1  
        # 開始分割 window
        current_time = start_time
        win_dfs = []
        for index in tqdm(range(total_windows), desc="Building time train wins"):
            # 建立窗口區間
            win_mask = (current_time <= semantic_logs_df["timestamp"]) & (semantic_logs_df["timestamp"] < (current_time + pd.Timedelta(seconds=win_size_secs)))
            win_df = semantic_logs_df[win_mask].copy()
            win_df["win_id"] = index
            win_df.sort_values(by=["lem_score"], ascending=False, inplace=True)
            win_dfs.append(win_df)
            current_time += pd.Timedelta(seconds=win_step_secs)
        return pd.concat(win_dfs, ignore_index=True)
    
    def build_count_wins(
        self, 
        df: pd.DataFrame,
        win_size_logs: int,
        win_step_logs: int,
    ) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.sort_values(by=["timestamp"], ascending=True, inplace=True)
        # 預估總共需要幾個 window
        total_wins = int(len(df) // win_step_logs) + 1 
        # 開始分割 window
        current_index = 0
        win_dfs = []
        for index in tqdm(range(total_wins), desc="Building count wins"):
            # 建立窗口區間
            win_df = df.iloc[current_index:current_index + win_size_logs].copy()
            win_df["win_id"] = index
            win_dfs.append(win_df)
            current_index += win_step_logs
        return pd.concat(win_dfs, ignore_index=True)

    def build_train_adllm_count_wins(
        self,
        logs_df: pd.DataFrame,
        lem_score_map: dict[str, float],
        win_size_logs: int,
        win_step_logs: int,
    ) -> pd.DataFrame:
        semantic_logs_df = semantic_logs_df.copy()
        wins_df = self.build_count_wins(semantic_logs_df, win_size_logs, win_step_logs)
        wins_df["binary_label"] = wins_df["label"].apply(lambda x: 0 if x == "-" else 1)
        wins_df["lem_score"] = wins_df["log"].apply(lambda x: lem_score_map[x])

        win_dfs = []
        for win_id, win_df in tqdm(wins_df.groupby("win_id"), desc="Building train adllm count wins"):
            win_df.sort_values(by=["lem_score"], ascending=False, inplace=True)
            win_dfs.append(win_df)
        return pd.concat(win_dfs, ignore_index=True)

        # semantic_logs_df = semantic_logs_df.copy()
        # semantic_logs_df["binary_label"] = semantic_logs_df["label"].apply(lambda x: 0 if x == "-" else 1)
        # semantic_logs_df["lem_score"] = semantic_logs_df["log"].apply(lambda x: lem_score_map[x])
        # semantic_logs_df["timestamp"] = pd.to_datetime(semantic_logs_df["timestamp"], unit="s")
        # semantic_logs_df.sort_values(by=["timestamp"], ascending=True, inplace=True)
        # # 預估總共需要幾個 window
        # total_wins = int(len(semantic_logs_df) // win_step_semantic_logs) + 1  
        # # 開始分割 window
        # current_index = 0
        # win_dfs = []
        # for index in tqdm(range(total_wins), desc="Building count train wins"):
        #     # 建立窗口區間
        #     win_df = semantic_logs_df.iloc[current_index:current_index + win_size_semantic_logs].copy()
        #     win_df["win_id"] = index
        #     win_df.sort_values(by=["lem_score"], ascending=False, inplace=True)
        #     win_dfs.append(win_df)
        #     current_index += win_step_semantic_logs
        # return pd.concat(win_dfs, ignore_index=True)

    def save_train_adllm_wins(self, df: pd.DataFrame, dtype: types.WinDatasetTypes, wtype: types.SlidingWindowTypes):
        df.to_csv(self.get_train_adllm_wins_path(dtype, wtype), index=False, chunksize=10000)
    
    def load_train_adllm_wins(self, dtype: types.WinDatasetTypes, wtype: types.SlidingWindowTypes) -> pd.DataFrame:
        return pd.read_csv(self.get_train_adllm_wins_path(dtype=dtype, wtype=wtype)).astype({
            "log": str, "lem_score": float, "binary_label": int,
        })
    
