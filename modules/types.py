
from pydantic import BaseModel
from typing import Literal

DatasetTypes = Literal["BGL", "Liberty", "Thunderbird", "test"]

SlidingWindowTypes = Literal["count", "time"]

SamplingTypes = Literal["our", "logllm"]

BaseLLMTypes = Literal["gemma-2-9b", "gemma-3-4b-it", "Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct"]

class EvaluationResult(BaseModel):
    accuracy: float
    f1: float
    precision: float
    recall: float
    cm: list[list[int]]

class AnomalyDetecteLLMSavePaths(BaseModel):
    lora: str

class LogEmbedModelSavePaths(BaseModel):
    bert: str
    classifier: str

class DatasetConfig(BaseModel):
    path: str
    fromat: str
    start_line: int
    end_line: int | None
    timestamp_column: str
    feat_columns: list[str]
    label_column: str

class WorkConfig(BaseModel):
    name: str
    system_name: str
    dataset_config: DatasetConfig
