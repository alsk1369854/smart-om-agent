import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import ast
from torch import nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Tuple
from modules import models, configs, datasets, recorders, types
from modules.utils import eval_utils, train_utils, plt_utils

def train_one_epoch(
    *,
    lem: models.LogEmbedModel, 
    train_loader: DataLoader,
    gradient_accumulation_steps: int,
    optimizer: torch.optim.AdamW,
    loss_fn: nn.BCEWithLogitsLoss,
    smoothing: float,
) -> Tuple[types.EvaluationResult, float]:
    lem.train()
    total_loss, count, step = 0, 0, 0
    all_preds, all_labels = [], []
    optimizer.zero_grad() # 初始化梯度

    loop = tqdm(train_loader, desc="Training")
    for batch in loop:
        step += 1
        inputs = batch["log"]
        labels = batch["label"].to(lem.device)

        smoothed_labels = train_utils.apply_label_smoothing_binary(labels, smoothing)

        outputs = lem(inputs)
        loss = loss_fn(outputs, smoothed_labels) # 計算 loss
        loss.backward() # 計算梯度

        # 梯度累積
        if step % gradient_accumulation_steps == 0:
            optimizer.step()  # 更新權重
            optimizer.zero_grad()  # 重置梯度

        # 更新紀錄
        preds = torch.sigmoid(outputs)
        preds = (preds >= 0.5).int()
        labels = (labels >= 0.5).int()
        all_preds += (preds.cpu().tolist())
        all_labels += (labels.cpu().tolist())

        total_loss += loss.item()
        count += labels.shape[0]
        avg_loss = total_loss / count
        loop.set_postfix(loss=f"{avg_loss:.4f}")
    
    optimizer.step()  # 更新權重
    optimizer.zero_grad()  # 重置梯度        
    result = eval_utils.evaluate_binary_classifier(preds=all_preds, labels=all_labels)
    
    return result, total_loss / count


def eval(
    *,
    lem: models.LogEmbedModel,
    test_loader: DataLoader,
    loss_fn: nn.BCEWithLogitsLoss,
    smoothing: float,
) -> Tuple[types.EvaluationResult, float]:
    lem.eval()
    total_loss, count = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"Evaluating")
        for batch in loop:
            inputs = batch["log"]
            labels = batch["label"].to(lem.device)

            smoothed_labels = train_utils.apply_label_smoothing_binary(labels, smoothing)

            outputs = lem(inputs)
            loss = loss_fn(outputs, smoothed_labels) # 計算 loss

            preds = torch.sigmoid(outputs)
            preds = (preds >= 0.5).int()
            labels = (labels >= 0.5).int()
            all_preds += (preds.cpu().tolist())
            all_labels += (labels.cpu().tolist())

            # 更新紀錄
            total_loss += loss.item()
            count += labels.shape[0]
            avg_loss = total_loss / count
            loop.set_postfix(loss=f"{avg_loss:.4f}")

    result = eval_utils.evaluate_binary_classifier(preds=all_preds, labels=all_labels)
    return result, total_loss / count


def train(
    *,
    lem: models.LogEmbedModel, 
    epochs: int,
    lr: float,
    gradient_accumulation_steps: int,
    train_loader: DataLoader, 
    test_loader: DataLoader,
    save_base: str,
) -> Tuple[types.EvaluationResult, float]:
    recorder = None 
    best_f1 = 0.0

    smoothing = 0.1
    optimizer = torch.optim.AdamW(lem.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=6, T_mult=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        print(f"Training Epoch {epoch+1}/{epochs}")
        
        # 訓練一個 epoch
        train_start_time = datetime.now()
        train_result, train_loss = train_one_epoch(
            lem=lem, 
            train_loader=train_loader, 
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            smoothing=smoothing,
        )
        train_end_time = datetime.now()
        
        # 評估當前模型
        eval_start_time = datetime.now()
        eval_result, eval_loss = eval(
            lem=lem,
            test_loader=test_loader,
            loss_fn=loss_fn,
            smoothing=smoothing,
        )
        eval_end_time = datetime.now()

        # 存儲最好的模型
        if eval_result.f1 > best_f1:
            best_f1 = eval_result.f1
            lem.save_pretrained(path=os.path.join(save_base, "best"))

        # 紀錄日誌
        record_dict = {
            "epoch": epoch+1,
            "train_mins": (train_end_time - train_start_time).seconds / 60,
            "eval_mins": (eval_end_time - eval_start_time).seconds / 60,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }
        for k, v in train_result.model_dump().items():
            record_dict[f"train_{k}"] = v

        for k, v in eval_result.model_dump().items():
            record_dict[f"eval_{k}"] = v
            
        if recorder is None:
            recorder = recorders.HistoryRecorder(save_base=save_base, fieldnames=list(record_dict.keys()))
        recorder.add_record(record_dict)
        history_df = pd.read_csv(os.path.join(save_base, "history.csv"))
        history_df["eval_cm"] = history_df["eval_cm"].apply(ast.literal_eval)
        plt_utils.save_train_history_png(history_df, os.path.join(save_base, "history.png"))

        # 打印評估
        eval_utils.print_evaluation_result(result=eval_result,  cm_labels=["Abnormal", "Normal"])

        scheduler.step()
        