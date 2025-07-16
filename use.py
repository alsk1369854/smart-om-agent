import re
import pandas as pd
from modules import models, agents

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
    
    # 1. Instantiate the log embedding model
    log_embed_model_path = "./output/TEST/lem/best"
    log_embed_model = models.LogEmbedModel.from_pretrained(path=log_embed_model_path)
    
    # 2. Instantiate the anomaly detection LLM
    base_llm_path = "./hf_models/gemma-2-9b"
    lora_path = "./output/TEST/adllm/gemma-2-9b/count_logllm_win/best"
    anomaly_detection_llm = models.AnomalyDetectionLLM.from_pretrained(
        save_path=lora_path,
        base_llm_path=base_llm_path,
        system_name="BlueGene/L supercomputer system",
        field_names="Level, Content",
    )
    
    # 3. Instantiate the Smart Operation And Maintenance Agent
    agent = agents.SmartOperationAndMaintenanceAgent(
        lem=log_embed_model,
        adllm=anomaly_detection_llm,
    )
    
    # Test the agent with some logs
    log_path="./data/BGL/BGL.log"
    log_fromat="<label> <timestamp> <date> <node> <time> <node_repeat> <type> <component> <level> <content>"
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
    for i, test_case in enumerate(test_cases):
        win, label = test_case
        output = agent.process_logs(win)
        print(f"{i+1}. True Label: {label}, Predicted System State: {output}")
    

if __name__ == "__main__":
    main()