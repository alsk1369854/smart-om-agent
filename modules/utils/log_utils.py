import pandas as pd
import re
from typing import  Callable, Optional, Callable
from tqdm import tqdm

LOG_REPLACE_PATTERN = "|".join([
    # Boolean：True / false
    r"True", r"true", r"False", r"false",
    
    # Number words：zero / one / two
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b",
    
    # Weekdays：Mon / Tue / Wed
    r"\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b",
    
    # Months and dates：Jan 10 / Feb 12
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b",
    
    # IP Address：192.168.0.1 / 192.168.0.1:8080
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?",
    
    # MAC Address：00:1A:2B:3C:4D:5E
    r"([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}",
    r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}",
    
    # File Path: /var/log/error.log
    r"[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*",
    
    # Hexadecimal：0x1234abcd
    r"\b[0-9a-fA-F]{8}\b",
    r"\b[0-9a-fA-F]{10}\b",
    
    # Email：user@example.com
    r"(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)",
    r"(\w+[\w\.]*)@(\w+[\w\.]*)",
    
    # Mixed Variables：eth0 / cpu1
    r"[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*",
])
LOG_REPLACE_REGEX = re.compile(LOG_REPLACE_PATTERN)
DOTS_REGEX = re.compile(r"[\.]{3,}")
def log_regex_replase(log: str) -> str:
    log = re.sub(DOTS_REGEX, ".. ", log)   # Replace multiple "." with ".. "
    log = re.sub(LOG_REPLACE_PATTERN, "<*>", log)
    return log

class LogDataFrameHelper:
    def __init__(self, bast_log_path: str):
        self.bast_log_path = bast_log_path

    def save_to_csv(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False, chunksize=10000)
        
    def load_struct_logs(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    
    def build_struct_logs(
        self,
        log_path: str,
        log_format: str, # e.g. "<label> <timestamp> <date> <node> <time> <node_repeat> <type> <component> <level> <content>",
        start_line: int = 0,
        end_line: Optional[int] = None,
    ) -> pd.DataFrame:
        columns = log_format.split(" ")
        columns = [f.strip("<>") for f in columns]
        columns_len = len(columns)
        data = []
        lines_count = -1
        with open(log_path, "r", encoding="latin-1") as f:
            while True:
                line = f.readline()
                lines_count += 1
                if not line or (lines_count >= end_line):
                    break
                if start_line > lines_count:
                    continue

                line.strip()
                cells = re.split(r"\s+", line)
                struct_line = cells[:columns_len-1] + [" ".join(cells[columns_len-1:]).strip()]
                data.append(struct_line)
        return pd.DataFrame(data, columns=columns)
    
    def build_semantic_logs(
        self,
        struct_logs_df: pd.DataFrame,
        feature_columns: list[str],
        log_regex_replase_fn: Callable[[str], str]
    ) -> pd.DataFrame:
        df = struct_logs_df.copy()
        data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building semantic logs"):
            timestamp = row["timestamp"]
            log = ", ".join(row[feature_columns])
            log = log_regex_replase_fn(log)
            label = row["label"]
            data.append([timestamp, log, label])
        return pd.DataFrame(data, columns=["timestamp", "log", "label"])

    def load_semantic_logs(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path).astype({"log": str, "timestamp": int, "label": str})
    
    def load_wins(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path).astype({ "timestamp": int, "win_id": int })
    
    def build_count_wins(
        self, 
        df: pd.DataFrame,
        win_size_logs: int = 100,  # 每個 window 的日誌數量 (數量)
        win_step_logs: int = 100,  # 每個 window 的步進 (數量)
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
    
    def build_time_wins(
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
        total_secs = (end_time - start_time).total_seconds()
        total_wins = int(total_secs // win_step_secs) + 1  
        # 開始分割 window
        current_time = start_time
        win_dfs = []
        for index in tqdm(range(total_wins), desc="Building time windows"):
            # 建立窗口區間
            win_mask = (current_time <= df["timestamp"]) & (df["timestamp"] < (current_time + pd.Timedelta(seconds=win_size_secs)))
            win_df = df[win_mask].copy()
            win_df["win_id"] = index
            win_dfs.append(win_df)
            current_time += pd.Timedelta(seconds=win_step_secs)
        return pd.concat(win_dfs, ignore_index=True)
    
    def load_train_adllm_wins(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path).astype({ 
            "log": str, "lem_score": float, "binary_label": int, "timestamp": int, "win_id": int
        })
    
    def build_train_adllm_wins(
        self,
        wins_df: pd.DataFrame,
        lem_score_map: dict[str, float],
    ) -> pd.DataFrame:
        wins_df = wins_df.copy()
        wins_df["binary_label"] = wins_df["label"].apply(lambda x: 0 if x == "-" else 1)
        wins_df["lem_score"] = wins_df["log"].apply(lambda x: lem_score_map[x])
        win_dfs = []
        for win_id, win_df in tqdm(wins_df.groupby("win_id"), desc="Building train adllm wins"):
            win_df.sort_values(by=["lem_score"], ascending=False, inplace=True)
            win_df["win_id"] = win_id
            win_dfs.append(win_df)
        return pd.concat(win_dfs, ignore_index=True)
    