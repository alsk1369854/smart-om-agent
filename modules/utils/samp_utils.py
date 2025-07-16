import pandas as pd
import numpy as np
from typing import Hashable, Tuple, TypeVar, Tuple
from random import Random
from collections import defaultdict
from .. import environments

T = TypeVar("T")
U = TypeVar("U", bound=Hashable)
def minority_oversampling(
    x: list[T],
    y: list[U],
    beta: float = 1,
    random_state: int = environments.RANDOM_STATE,
) -> Tuple[list[T], list[U]]:
    assert len(x) == len(y)
    random = Random(random_state)

    y_indexes = defaultdict(list)
    for i, label in enumerate(y):
        y_indexes[label].append(i)

    max_label_count = max([len(v) for k, v in y_indexes.items()])
    print(f"最大類別數量 {max_label_count} 筆")

    samp_x = x.copy()
    samp_y = y.copy()
    for label, indexes in y_indexes.items():
        label_count = len(indexes)
        if max_label_count == label_count:
            continue
        
        oversamp_num = int((max_label_count - label_count) * beta)
        oversamp_indexes = [random.choice(indexes) for _ in range(oversamp_num)]

        samp_x += [x[index] for index in oversamp_indexes]
        samp_y += [y[index] for index in oversamp_indexes]

        print(f"已過採樣類別 '{label}' : 新增 {oversamp_num} 筆")

    # 整體隨機打亂樣本
    combined = list(zip(samp_x, samp_y))
    random.shuffle(combined)
    shuffled_x, shuffled_y = zip(*combined)

    return list(shuffled_x), list(shuffled_y)


def logllm_minority_oversampling(
    x: list[T],
    y: list[U], 
    alpha: float = 0.3,
    random_state: int = environments.RANDOM_STATE,
) -> Tuple[list[T], list[U]]:
    assert len(x) == len(y)
    random = Random(random_state)
    
    label_indexes = defaultdict(list)
    for i, label in enumerate(y):
        label_indexes[label].append(i)
    
    total_label_count = len(y)
    less_label = list(label_indexes.keys())[0]
    majority_label = list(label_indexes.keys())[0]
    for label, indexes in label_indexes.items():
        label_len = len(indexes)
        if len(label_indexes[less_label]) > label_len:
            less_label = label
        if len(label_indexes[majority_label]) < label_len:
            majority_label = label

            
    if (len(label_indexes[less_label]) / total_label_count) >= alpha:
        return x, y
    
    oversamp_num = int(alpha * len(label_indexes[majority_label]) / (1 - alpha))
    oversamp_indexes = [random.choice(label_indexes[less_label]) for _ in range(oversamp_num)]

    samp_x = x.copy()
    samp_y = y.copy()
    samp_x += [x[index] for index in oversamp_indexes]
    samp_y += [y[index] for index in oversamp_indexes]
    print(f"最大類別 '{majority_label}' 數量 {len(label_indexes[majority_label])} 筆")
    print(f"已過採樣類別 '{less_label}' : 新增 {oversamp_num} 筆")
    
    # 整體隨機打亂樣本
    combined = list(zip(samp_x, samp_y))
    random.shuffle(combined)
    shuffled_x, shuffled_y = zip(*combined)

    return list(shuffled_x), list(shuffled_y)


def train_test_sampling(
    df: pd.DataFrame, 
    train_ratio: float, 
    label_column: str = "label",
    random_state: int = environments.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_len = len(df)
    
    # Our 根據常態分布對各類別隨機採樣
    label_counts = df[label_column].value_counts(normalize=True)
    train_label_samp_nums = (label_counts * df_len * train_ratio).apply(np.ceil).astype(int)

    train_idx = []
    test_idx = []

    grouped = df.groupby(label_column)
    for label, group in grouped:
        group = group.sample(frac=1, random_state=random_state)  # 隨機打亂每類別資料
        samp_num = min(train_label_samp_nums[label], len(group))
        train_idx.extend(group.index[:samp_num])
        test_idx.extend(group.index[samp_num:])

    # 回復原始順序
    train_df = df.loc[train_idx].sort_index().reset_index(drop=True)
    test_df = df.loc[test_idx].sort_index().reset_index(drop=True)
    return train_df, test_df


def logllm_train_test_sampling(
    df: pd.DataFrame,
    train_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_len = int(len(df) * environments.TRAIN_RATIO)
    train_logllm_logs_df = df[:train_len]
    test_logllm_logs_df = df[train_len:].reset_index(drop=True)
    return train_logllm_logs_df, test_logllm_logs_df
