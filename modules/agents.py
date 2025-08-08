from modules import models
from modules.utils import log_utils
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing import Callable

class SmartOperationAndMaintenanceAgent():
    TOP_K_ABNORMAL_LOGS = 5
    LOG_EMBEDDING_BATCH_SIZE = 100
    
    lem: models.LogEmbedModel
    adllm: models.AnomalyDetectionLLM
    log_regex_replace_func: Callable[[str], str]
    graph: CompiledStateGraph

    def __init__(
        self, *,
        lem: models.LogEmbedModel,
        adllm: models.AnomalyDetectionLLM,
        log_regex_replace_func: Callable[[str], str] = log_utils.log_regex_replace,
    ):
        self.lem = lem
        self.adllm = adllm
        self.log_regex_replace_func = log_regex_replace_func
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:

        class State(BaseModel):
            logs: list[str] = Field(default_factory=list)
            system_state: str = Field(default="")

        def log_semantic_extraction_node(state: State):
            logs = state.logs
            logs = [self.log_regex_replace_func(log) for log in logs]
            return { "logs": logs }     

        def remove_duplicate_semantics_node(state: State):
            logs = state.logs
            logs = list(set(logs))
            return { "logs": logs }     
            
        def log_embedding_and_abnormal_ranking_node(state: State):
            logs = state.logs
            scores = []
            for i in range(0, len(logs), self.LOG_EMBEDDING_BATCH_SIZE):
                batch = logs[i:i + self.LOG_EMBEDDING_BATCH_SIZE]
                scores.extend(self.lem.pred(batch))
            
            sorted_logs = sorted(zip(logs, scores), key=lambda x: x[1], reverse=True)
            logs = [log for log, score in sorted_logs]
            return { "logs": logs }     

        def prompt_building_ang_anomaly_detection_node(state: State):
            logs = state.logs
            if len(logs) > self.TOP_K_ABNORMAL_LOGS:
                logs = logs[:self.TOP_K_ABNORMAL_LOGS]
            system_state = self.adllm.generate(logs)
            return { "logs": logs, "system_state": system_state }

        workflow = StateGraph(State)
        # Add nodes to the workflow
        workflow.add_node("log_semantic_extraction", log_semantic_extraction_node)
        workflow.add_node("remove_duplicate_semantics", remove_duplicate_semantics_node)
        workflow.add_node("log_embedding_and_abnormal_ranking", log_embedding_and_abnormal_ranking_node)
        workflow.add_node("prompt_building_ang_anomaly_detection", prompt_building_ang_anomaly_detection_node)
        # Connect the nodes
        workflow.add_edge(START, "log_semantic_extraction")
        workflow.add_edge("log_semantic_extraction", "remove_duplicate_semantics")
        workflow.add_edge("remove_duplicate_semantics", "log_embedding_and_abnormal_ranking")
        workflow.add_edge("log_embedding_and_abnormal_ranking", "prompt_building_ang_anomaly_detection")
        workflow.add_edge("prompt_building_ang_anomaly_detection", END)
        return workflow.compile()    

    def process_logs(self, logs: list[str]) -> str:
        result = self.graph.invoke({ "logs": logs })
        system_state = result["system_state"]
        return system_state