import torch
import pandas as pd
from collections import Counter


def apply_label_smoothing_binary(targets: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    # 對正確標籤 1 進行平滑
    smoothed_target = targets * (1 - smoothing) + 0.5 * smoothing
    return smoothed_target


def parse_wins_df(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[list[list[str]], list[str]]:
    wins = []
    labels = []
    for _, group in df.groupby("win_id"):
        if group.empty:
            continue
        label = group["binary_label"].max()
        if drop_duplicates:
            group.drop_duplicates(subset=["log"], inplace=True)
        logs = group["log"].tolist()
        wins.append(logs)
        labels.append(label)
    return wins, labels


LOG_EMBED_MODEL_TRAINING_INFO_TEMPLATE = """
# Log Embed Model Training information
train_case:             {train_case_name}
epochs:                 {epochs}
lr:                     {lr}
batch_size:             {batch_size}
oversamp_beta:          {oversamp_beta}
total_label:            {total_label}
abnormal_label_rate:    {abnormal_label_rate}
normal_label_rate:      {normal_label_rate}
"""
def print_log_embed_model_training_info(
    train_case_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    oversamp_beta: float,
    train_labels: list[str],
) -> None:
    label_counter = Counter(train_labels)
    total_labels = len(train_labels)
    abnormal_rate = f"{label_counter[1]}({label_counter[1] / total_labels * 100:.1f}%)"
    normal_rate = f"{label_counter[0]}({label_counter[0] / total_labels * 100:.1f}%)"
    print(LOG_EMBED_MODEL_TRAINING_INFO_TEMPLATE.format_map({
        "train_case_name": train_case_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "oversamp_beta": oversamp_beta,
        "total_label": total_labels,
        "abnormal_label_rate": abnormal_rate,
        "normal_label_rate": normal_rate,
    }))


ANOMALY_DETECTION_LLM_TRAINING_INFO_TEMPLATE = """
# Anomaly Detection LLM Training information
train_case:     {train_case_name}
base_llm_name:  {base_llm_name}
win_type:       {win_type}
epochs:         {epochs}
lr:             {lr}
top_k_logs:     {top_k_logs}
batch_size:     {batch_size}
oversamp_beta:  {oversamp_beta}
total_label:            {total_label}
abnormal_label_rate:    {abnormal_label_rate}
normal_label_rate:      {normal_label_rate}
"""
def print_anomaly_detection_llm_training_info(
    train_case_name: str,
    base_llm_name: str,
    win_type: str,
    epochs: int,
    lr: float,
    top_k_logs: int,
    batch_size: int,
    oversamp_beta: float,
    train_labels: list[str],
) -> None:
    label_counter = Counter(train_labels)
    total_labels = len(train_labels)
    abnormal_rate = f"{label_counter[1]}({label_counter[1] / total_labels * 100:.1f}%)"
    normal_rate = f"{label_counter[0]}({label_counter[0] / total_labels * 100:.1f}%)"
    print(ANOMALY_DETECTION_LLM_TRAINING_INFO_TEMPLATE.format_map({
        "train_case_name": train_case_name,
        "base_llm_name": base_llm_name,
        "win_type": win_type,
        "epochs": epochs,
        "lr": lr,
        "top_k_logs": top_k_logs,
        "batch_size": batch_size,
        "oversamp_beta": oversamp_beta,
        "total_label": total_labels,
        "abnormal_label_rate": abnormal_rate,
        "normal_label_rate": normal_rate,
    }))
    