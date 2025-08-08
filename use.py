import re
import pandas as pd
from modules import models, agents, configs, environments, types
from modules.utils import log_utils


def main() -> None:
    # Example usage of Smart Operation And Maintenance Agent
    CASE_NAME: str = "bgl-cw-gemma2-9b" # customize your case name
    DATASET_TYPE: types.DatasetTypes = "test" # "BGL" | "Liberty" | "Thunderbird"
    BASE_LLM: types.BaseLLMTypes = "gemma-2-9b" # "gemma-2-9b" | "gemma-3-4b-it" | "Llama-3.1-8B-Instruct" | "Llama-3.2-3B-Instruct"
    
    work_config = configs.WORK_CONFIG_MAP[DATASET_TYPE]
    
    # 1. Instantiate the log embedding model
    log_embed_model = models.LogEmbedModel.from_pretrained(save_path=f"./output/{CASE_NAME}/lem/best")
    
    # 2. Instantiate the anomaly detection LLM
    anomaly_detection_llm = models.AnomalyDetectionLLM.from_pretrained(
        save_path=f"./output/{CASE_NAME}/adllm/best",
        base_llm_path=f"./hf_models/{BASE_LLM}",
        system_name=work_config.system_name,
        field_names=", ".join([col.title() for col in work_config.dataset_config.feat_columns]),
    )
    
    # 3. Instantiate the Smart Operation And Maintenance Agent
    agent = agents.SmartOperationAndMaintenanceAgent(
        lem=log_embed_model,
        adllm=anomaly_detection_llm,
    )
    
    # ===== Test the agent with some logs =====
    ldfh = log_utils.LogDataFrameHelper()
    log_struct_df = ldfh.load_struct_logs(f"{work_config.dataset_config.path}-struct-logs.csv")

    # Prepare the logs and labels for testing
    test_log_struct_df = log_struct_df.iloc[int(len(log_struct_df)*environments.TRAIN_RATIO):] 
    logs = test_log_struct_df[["level", "content"]].apply(lambda x: f"{x['level']}, {x['content']}", axis=1).tolist()
    labels = test_log_struct_df["label"].apply(lambda x: "Normal" if x == "-" else "Abnormal").tolist()

    # Create test cases
    test_cases: list[tuple[list[str], str]] = []
    for i in range(0, len(logs), 100):
        win = logs[i:i+100]
        label = "Abnormal" if "Abnormal" in set(labels[i:i+100]) else "Normal"
        test_cases.append((win, label))

    # Process each test case
    correct_count = 0
    for i, test_case in enumerate(test_cases):
        win, label = test_case
        output = agent.process_logs(win)
        if output == label:
            correct_count += 1
        print(f"{i+1:6d}  Label: {label:<10}  Predicted: {output:<10}  Correct: {output == label}")
    print(f"Total correct predictions: {correct_count}/{len(test_cases)}({correct_count/len(test_cases)*100:.2f}%)")

if __name__ == "__main__":
    main()