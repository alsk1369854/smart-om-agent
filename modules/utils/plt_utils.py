import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from typing import Tuple, Tuple

def draw_label_distribution(
    labels: list[str],
    title: str = "Distribution of Labels", 
    figsize: None | Tuple[int, int] = None,
    horizontal: bool = False
) -> None:
    labels_len = len(labels)
    # 計算 Label 欄位的分佈
    label_counter = Counter(labels)
    
    label_list = list(label_counter.keys())
    label_list.sort()
    count_list = [label_counter[label] for label in label_list]
    
    # 設定圖形大小
    plt.figure(figsize=(len(label_list) * 1.5, 5) if figsize is None else figsize)
    
    # 繪製柱狀圖
    if horizontal:
        plt.barh(label_list, count_list, color="skyblue")
        # 在 bar 上方標示數值
        for i, count in enumerate(count_list):
            plt.text(count + max(count_list) * 0.01, i, f"{str(count)}({count / labels_len * 100:.1f}%)", 
                        va='center', fontsize=8)
        plt.title(title, fontsize=14)
        plt.xlabel("Count", fontsize=12)
        plt.ylabel("Label", fontsize=12)
        plt.ylim(-0.5, len(count_list) - 0.5)  # 確保 x 軸範圍適配所有標籤
        plt.tight_layout()
        plt.show()
        
    else:
        plt.bar(label_list, count_list, color="skyblue")

        # 在 bar 上方標示數值
        for i, count in enumerate(count_list):
            plt.text(i, count + max(count_list) * 0.01, f"{str(count)}({count / labels_len * 100:.1f}%)", 
                        ha='center', va='bottom', fontsize=8)

        # 添加標題與標籤
        plt.title(title, fontsize=14)
        plt.xlabel("Label", fontsize=12)
        plt.ylabel("Count", fontsize=12)

        plt.xlim(-0.5, len(label_list) - 0.5)  # 確保 x 軸範圍適配所有標籤
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    
def set_label_distribution_plot(
    ax: plt.Axes, 
    labels: list[str],
    title: str = "Label Distribution"
) -> None:
    labels_len = len(labels)
    label_counter = Counter(labels)
    label_list = list(label_counter.keys())
    label_list.sort()
    count_list = [label_counter[label] for label in label_list]
    
    # 設定圖形大小
    ax.bar(label_list, count_list, color="skyblue")
    for i, count in enumerate(count_list):
        ax.text(i, count + max(count_list) * 0.01, f"{str(count)}\n({count / labels_len * 100:.1f}%)", 
            ha='center', va='bottom', fontsize=8)

        # 添加標題與標籤
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Label", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlim(-0.5, len(label_list) - 0.5)
    ax.set_ylim(0, max(count_list) + 0.1 * max(count_list))
    # ax.set_xticks(rotation=45)
    # 設定標籤旋轉角度
    ax.tick_params(axis='x', rotation=45)
    # ax.tick_params(axis='y', rotation=0)
    
    # # Loss vs Epoch
    # color = "tab:red"
    # ax.plot(epoch_data, loss_data, marker="o", color=color, linestyle='-', label="Loss")
    # ax.set_title(title)
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Loss")
    # ax.legend()
    # ax.grid(True)
    

def set_epoch_loss_plot(ax: plt.Axes, epoch_data: list[int], loss_data: list[float], title: str = "Train Loss vs Epoch") -> None:
    # Loss vs Epoch
    color = "tab:red"
    ax.plot(epoch_data, loss_data, marker="o", color=color, linestyle='-', label="Loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    
    
def set_epoch_eval_metrics_plot(
    ax: plt.Axes, 
    epoch_data: list[int],
    f1_data: list[float],
    precision_data: list[float],
    recall_data: list[float],
    title: str = "Eval Metrics vs Epoch",
    show_best_annot_box: bool = True, 
) -> None:
    # 顏色設置
    f1_color = 'tab:orange'
    precision_color = 'tab:purple'
    recall_color = 'tab:brown'

    # 繪製所有指標曲線
    ax.plot(epoch_data, f1_data, marker="o", color=f1_color, linestyle='-', label="F1-Score")
    ax.plot(epoch_data, precision_data, marker="o", color=precision_color, linestyle='--', label="Precision")
    ax.plot(epoch_data, recall_data, marker="o", color=recall_color, linestyle='-.', label="Recall")

    # 標註最高 F1 Score 的 epoch
    best_index = f1_data.index(max(f1_data))
    best_epoch = epoch_data[best_index]
    best_f1 = f1_data[best_index]
    best_precision = precision_data[best_index]
    best_recall = recall_data[best_index]

    # 組合成多行文字
    if show_best_annot_box:
        annot_text = (
            "Best F1-Score\n"
            f"Epoch: {best_epoch}\n"
            f"F1: {best_f1:.4f}\n"
            f"Prec: {best_precision:.4f}\n"
            f"Recall: {best_recall:.4f}"
        )
        ax.annotate(
            annot_text,
            xy=(best_epoch, best_f1),
            xytext=(0, -10),  # 向右 0px，下移 10px，剛好在 legend 下方
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=9,
            color=f1_color,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=f1_color, facecolor='white', alpha=0.8)
        )

    # 圖表設置
    ax.set_title(title)
    ax.set_xlabel("Epoch", labelpad=None)
    ax.set_ylabel("Score")
    ax.legend(loc='lower right')
    ax.grid(True)


def set_confusion_matrix_plot(
    ax: plt.Axes,
    cm_data: list[list[int]],
    title: str = "Confusion Matrix",
    labels: list[str] = ["Abnormal", "Normal"],
) -> None:
    matrix_labels = [["TP", "FN"], ["FP", "TN"]]
    # 標註 TP/FN/FP/TN 和數字
    annotations = [[f"{matrix_labels[r][c]}\n{cm_data[r][c]}" for c in range(2)] for r in range(2)]
    sns.heatmap(
        cm_data, 
        annot=annotations, 
        fmt="", 
        cbar=False,
        xticklabels=labels,
        yticklabels=labels, 
        ax=ax, 
        cmap="Blues"
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    
def set_computational_cost_plot(
    ax: plt.Axes,
    epoch_data: list[int],
    train_mins_data: list[float],
    eval_mins_data: list[float],
    title: str = "Computational Cost vs Epoch",
) -> None:
    # 計算成本 vs Epoch
    total_color = 'tab:orange'
    train_color = 'tab:purple'
    eval_color = 'tab:brown'
    train_hours_data = [x / 60 for x in train_mins_data]
    eval_hours_data = [x / 60 for x in eval_mins_data]
    total_hours_data = list(map(lambda x, y: x + y, train_hours_data, eval_hours_data))
    ax.plot(epoch_data, total_hours_data, marker="o", color=total_color, linestyle='-', label="Total Cost")
    ax.plot(epoch_data, train_hours_data, marker="o", color=train_color, linestyle='--', label="Train Cost")
    ax.plot(epoch_data, eval_hours_data, marker="o", color=eval_color, linestyle='-.', label="Eval Cost")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hours")
    # 設定 y 軸最多顯示三位小數
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.legend()
    ax.grid(True)

def get_train_history_png(history_df: pd.DataFrame) -> plt.Figure:
    # 紀錄圖
    epoch_data = history_df["epoch"].tolist()
    f1_data = history_df["eval_f1"].tolist()
    best_index = f1_data.index(max(f1_data))
    best_cm = history_df["eval_cm"].tolist()[best_index]
    train_mins_data = history_df["train_mins"].tolist()
    eval_mins_data = history_df["eval_mins"].tolist()
    for i in range(1, len(train_mins_data)):
        train_mins_data[i] += train_mins_data[i - 1]
        eval_mins_data[i] += eval_mins_data[i - 1]
    
    fig, axes = plt.subplots(1, 4, figsize=(6*4, 5))
    set_epoch_loss_plot(
        axes[0], 
        epoch_data, 
        history_df["train_loss"].tolist()
    )
    set_epoch_eval_metrics_plot(
        axes[1],
        epoch_data,
        history_df["eval_f1"].tolist(),
        history_df["eval_precision"].tolist(),
        history_df["eval_recall"].tolist(),
    )
    set_confusion_matrix_plot(
        axes[2], 
        best_cm, 
        title="Best F1-Score Confusion Matrix"
    )
    set_computational_cost_plot(
        axes[3], 
        epoch_data, 
        train_mins_data, 
        eval_mins_data,
    )
    return fig

def save_train_history_png(history_df: pd.DataFrame, save_path: str) -> plt.Figure:
    fig = get_train_history_png(history_df)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig