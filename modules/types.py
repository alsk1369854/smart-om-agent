
import pandas as pd
from pydantic import BaseModel
from typing import Callable, Literal, Optional, Union


EvalTypes = Literal["without_stage_1_and_2", "without_stage_1", "without_stage_2", "smart_om_agent"]

DatasetTypes = Literal["train", "test", "train_logllm", "test_logllm"]

WinDatasetTypes =  Union[DatasetTypes, Literal["untrained_lem_train", "untrained_lem_test"]]

SlidingWindowTypes = Literal["count", "time"]

class EvaluationResult(BaseModel):
    accuracy: float
    f1: float
    precision: float
    recall: float
    cm: list[list[int]]

class AnomalyDetecteLLMSavePaths(BaseModel):
    llm: str

class LogEmbedModelSavePaths(BaseModel):
    bert: str
    classifier: str

class LogConfig(BaseModel):
    path: str
    fromat: str
    start_line: int
    end_line: int | None
    timestamp_column: str
    feat_columns: list[str]
    label_column: str

class TrainConfig(BaseModel):
    hf_models_path: str
    save_base: str 

class WorkConfig(BaseModel):
    name: str
    system_name: str
    log_config: LogConfig
    train_config: TrainConfig
