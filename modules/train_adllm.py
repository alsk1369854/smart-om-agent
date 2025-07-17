import os
import torch
import pandas as pd
import ast
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from modules import models, recorders, types
from modules.utils import train_utils, plt_utils

def train_one_epoch(
    *,
    adllm: models.AnomalyDetectionLLM, 
    train_loader: DataLoader,
    gradient_accumulation_steps: int,
    optimizer: torch.optim.AdamW,
    top_k_logs: int,
) -> float:
    adllm.train()
    total_loss, count, step = 0, 0, 0
    optimizer.zero_grad() # 初始化梯度

    loop = tqdm(train_loader, desc="Training")
    for batch in loop:
        step += 1
        log_wins = batch["log_win"]
        log_wins = [log_win[:top_k_logs] for log_win in log_wins]
        targets = batch["label"]
        
        # 訓練 adllm
        loss = adllm(log_wins=log_wins, targets=targets).loss
        loss.backward() # 計算梯度

        # 梯度累積
        if step % gradient_accumulation_steps == 0:
            optimizer.step()  # 更新權重
            optimizer.zero_grad()  # 重置梯度

        total_loss += loss.item()
        count += len(targets)
        avg_loss = total_loss / count
        loop.set_postfix(loss=f"{avg_loss:.4f}")
    
    optimizer.step()  # 更新權重
    optimizer.zero_grad()  # 重置梯度        
    
    return total_loss / count


def eval(
    *,
    adllm: models.AnomalyDetectionLLM, 
    test_loader: DataLoader,
    top_k_logs: int,
) -> types.EvaluationResult:
    adllm.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"Evaluating")
        for batch in loop:
            log_wins = batch["log_win"]
            log_wins = [log_win[:top_k_logs] for log_win in log_wins]
            targets = batch["label"]

            # 訓練 adllm
            generations = [adllm.generate(log_win=log_win) for log_win in log_wins]
            preds = [1 if generation == "Abnormal" else 0 for generation in generations]
            all_preds += preds
            all_labels += targets

    return train_utils.evaluate_binary_classifier(preds=all_preds, labels=all_labels)


def train(
    *,
    adllm: models.AnomalyDetectionLLM,
    epochs: int,
    lr: float,
    gradient_accumulation_steps: int,
    train_loader: DataLoader, 
    test_loader: DataLoader,
    top_k_logs: int,
    save_base: str,
) -> None:
    recorder = None 
    best_f1 = 0

    optimizer = torch.optim.AdamW(adllm.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=1)
    for epoch in range(epochs):
        print(f"Training Epoch {epoch+1}/{epochs}")

        # 訓練一個 epoch
        train_start_time = datetime.now()        
        train_loss = train_one_epoch(
            adllm=adllm,
            train_loader=train_loader, 
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            top_k_logs=top_k_logs,
        )
        train_end_time = datetime.now()
        
        # 評估當前模型
        eval_start_time = datetime.now()
        eval_result = eval(
            adllm=adllm,
            test_loader=test_loader,
            top_k_logs=top_k_logs,
        )
        eval_end_time = datetime.now()

        # 存儲最好的模型
        if eval_result.f1 > best_f1:
            best_f1 = eval_result.f1
            adllm.save_pretrained(path=os.path.join(save_base, "best"))

        # 紀錄日誌
        record_dict = {
            "epoch": epoch+1,
            "train_mins": (train_end_time - train_start_time).seconds / 60,
            "eval_mins": (eval_end_time - eval_start_time).seconds / 60,
            "train_loss": train_loss,
        }
        for k, v in eval_result.model_dump().items():
            record_dict[f"eval_{k}"] = v
            
        if recorder is None:
            recorder = recorders.HistoryRecorder(save_base=save_base, fieldnames=list(record_dict.keys()))
        recorder.add_record(record_dict)
        history_df = pd.read_csv(os.path.join(save_base, "history.csv"))
        history_df["eval_cm"] = history_df["eval_cm"].apply(ast.literal_eval)
        plt_utils.save_train_history_png(history_df, os.path.join(save_base, "history.png"))
        
        # 打印評估
        train_utils.print_evaluation_result(result=eval_result,  cm_labels=["Abnormal", "Normal"])

        scheduler.step()

