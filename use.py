import re
import pandas as pd
from modules import models, agents, configs, environments
from modules.utils import log_utils

def get_first_k_logs(log_path: str, log_fromat: str, k: int) -> pd.DataFrame:
    columns = [col.strip('<>') for col in log_fromat.split(' ')]
    data = []
    with open(log_path, 'r', encoding='latin-1') as f:
        for _ in range(k):
            line = f.readline()
            if not line:
                break
            line.strip()
            cells = re.split(r'\s+', line)
            struct_line = cells[:len(columns)-1] + [" ".join(cells[len(columns)-1:]).strip()]
            data.append(struct_line)
        
    return pd.DataFrame(data, columns=columns)

def main() -> None:
    # Example usage of Smart Operation And Maintenance Agent

    work_config = configs.TEST_WORK_CONFIG
    train_case_name = "TEST-count-wins-based-gemma-2-9b-with-our-oversampling"
    base_llm_name = "gemma-2-9b" # ./hf_models/{base_llm_name}

    # 1. Instantiate the log embedding model
    log_embed_model_path = f"./output/{train_case_name}/lem/best"
    log_embed_model = models.LogEmbedModel.from_pretrained(path=log_embed_model_path)
    
    # 2. Instantiate the anomaly detection LLM
    base_llm_path = f"./hf_models/{base_llm_name}"
    lora_path = f"./output/{train_case_name}/adllm/best"
    anomaly_detection_llm = models.AnomalyDetectionLLM.from_pretrained(
        save_path=lora_path,
        base_llm_path=base_llm_path,
        system_name=work_config.system_name,
        field_names=", ".join([col.title() for col in work_config.log_config.feat_columns]),
    )
    
    # 3. Instantiate the Smart Operation And Maintenance Agent
    agent = agents.SmartOperationAndMaintenanceAgent(
        lem=log_embed_model,
        adllm=anomaly_detection_llm,
    )
    
    # ===== Test the agent with some logs =====
    ldfh = log_utils.LogDataFrameHelper(work_config.log_config.path)
    log_struct_df = ldfh.load_struct()
    log_struct_df = log_struct_df.iloc[len(log_struct_df) * environments.TRAIN_RATIO:]  # Use the first half for testing

    log_path=work_config.log_config.path
    log_fromat=work_config.log_config.fromat
    logs_df = get_first_k_logs(log_path, log_fromat, 20000)
    logs_df = logs_df.iloc[15000:]
    logs = logs_df[["level", "content"]].apply(lambda x: f"{x['level']}, {x['content']}", axis=1).tolist()
    labels = logs_df["label"].apply(lambda x: "Normal" if x == "-" else "Abnormal").tolist()
    test_cases = []
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
        print(f"{i+1}. Label: {label}, Predicted: {output}, Correct: {output == label}")
    print(f"Total correct predictions: {correct_count}/{len(test_cases)}({correct_count/len(test_cases)*100:.2f}%)")

if __name__ == "__main__":
    main()