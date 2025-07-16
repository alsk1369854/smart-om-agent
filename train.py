
import torch
import gc
from torch.utils.data import DataLoader
from modules import configs, environments, models, datasets, train_lem, types, train_adllm
from modules.utils import log_utils, samp_utils, train_utils
from functools import partial


def get_log_feature_vector_map(
    train_case_name: str,
    logs: list[str],
) -> dict[str, float]:
    logs = list(set(logs))
    log_embed_model = models.LogEmbedModel.from_pretrained(path=f"./output/{train_case_name}/lem/best")
    batch_size: int = 256
    
    log_vector_map = {}
    for i in range(0, len(logs), batch_size):
        batch_logs = logs[i:i+batch_size]
        log_vectors = log_embed_model.pred(batch_logs)
        log_vector_map.update(dict(zip(batch_logs, log_vectors)))
        
    del log_embed_model
    gc.collect()
    torch.cuda.empty_cache()
    return log_vector_map


def stage_one_train(
    *,
    train_case_name: str,
    train_cases: tuple[str, str],
    test_cases: tuple[str, str],
    epochs: int,
    lr: float,
    safe_batch_size: int,
) -> None:
    """Stage one training for Log Embed Model."""
    train_logs, train_labels = train_cases
    test_logs, test_labels = test_cases
    
    # Oversamping
    oversamp_beta = 1 / (len(set(train_labels)) - 1)
    train_logs_oversmap, train_labels_oversmap = samp_utils.minority_oversampling(
        x=train_logs, y=train_labels, beta=oversamp_beta,
    )
    train_labels_oversmap = [0 if x == "-" else 1 for x in train_labels_oversmap]
    test_labels = [0 if x == "-" else 1 for x in test_labels]
    train_dataset = datasets.LogDataset(logs=train_logs_oversmap, labels=train_labels_oversmap)
    test_dataset = datasets.LogDataset(logs=test_logs, labels=test_labels)
    
    # Train setting
    batch_size = max(1, int(len(train_labels_oversmap) * 0.005))
    micro_batch_size = min(safe_batch_size, batch_size)
    gradient_accumulation_steps = batch_size // micro_batch_size
    train_utils.print_log_embed_model_training_info(
        train_case_name=train_case_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        oversamp_beta=oversamp_beta,
        train_labels=train_labels_oversmap,
    )
    
    # New Log Embed Model
    bert_path = "./hf_models/bert-base-uncased"
    log_embed_model = models.LogEmbedModel.new(bert_path=bert_path)

    # Training
    train_generator = torch.Generator().manual_seed(environments.RANDOM_STATE)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=micro_batch_size, generator=train_generator,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    train_lem.train(
        lem=log_embed_model, 
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_base=f"./output/{train_case_name}/lem"
    )
    
    del log_embed_model
    gc.collect()
    torch.cuda.empty_cache()


def stage_two_train(
    *,
    train_case_name: str,
    train_cases: tuple[str, str],
    test_cases: tuple[str, str],
    epochs: int,
    lr: float,
    safe_batch_size: int,
    top_k_logs: int,
    base_llm_name: str,
    sliding_window_type: types.SlidingWindowTypes,
    system_name: str,
    log_table_columns: list[str],
) -> None:
    """Stage two training for Anomaly Detection LLM."""
    train_wins, train_wins_label = train_cases
    test_wins, test_wins_label = test_cases
    
    # Oversamping
    oversamp_beta = 1
    train_wins_oversampe, train_labels_oversampe = samp_utils.minority_oversampling(
        x=train_wins, y=train_wins_label, beta=oversamp_beta, 
    )
    train_dataset = datasets.LogWindowDataset(log_wins=train_wins_oversampe, labels=train_labels_oversampe)
    test_dataset = datasets.LogWindowDataset(log_wins=test_wins, labels=test_wins_label)
    
    # Train setting
    batch_size = max(1, int(len(train_labels_oversampe) * 0.005))
    micro_batch_size = min(safe_batch_size, batch_size)
    gradient_accumulation_steps = batch_size // micro_batch_size
    train_utils.print_anomaly_detection_llm_training_info(
        train_case_name=train_case_name,
        base_llm_name=base_llm_name,
        win_type=sliding_window_type,
        epochs=epochs,
        lr=lr,
        top_k_logs=top_k_logs,
        batch_size=batch_size,
        oversamp_beta=oversamp_beta,
        train_labels=train_labels_oversampe,
    )
    
    # New Anomaly Detection LLM
    base_llm_path = f"./hf_models/{base_llm_name}"
    anomaly_detection_llm = models.AnomalyDetectionLLM.new(
        base_llm_path=base_llm_path, 
        system_name=system_name, 
        field_names=", ".join([col.title() for col in log_table_columns])
    )

    # Training
    train_generator = torch.Generator().manual_seed(environments.RANDOM_STATE)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=micro_batch_size,
        generator=train_generator,
        collate_fn=train_dataset.collate_fu
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=test_dataset.collate_fu)
    train_adllm.train(
        adllm=anomaly_detection_llm,
        epochs=epochs,
        lr=lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_loader=train_loader,
        test_loader=test_loader,
        top_k_logs=top_k_logs,
        save_base=f"./output/{train_case_name}/adllm" 
    )
    
    del anomaly_detection_llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    train_case_name = "smart-om-agent-bgl-cw-gemma2-9b"
    # Train output directory
    # - Log Embed Model: ./output/{train_case_name}/lem/
    # - Anomaly Detection LLM: ./output/{train_case_name}/adllm/
    
    # ===== Start of settings =====
    dataset_type: types.DatasetTypes = "test" # "BGL" | "Liberty" | "Thunderbird"
    sampling_type: types.SamplingTypes = "our" # "our" | "logllm"
    sliding_window_type: types.SlidingWindowTypes = "count" # "count" | "time"
    base_llm_name: types.BaseLLMTypes = "gemma-2-9b" # "gemma-2-9b" | "gemma-3-4b-it" | "Llama-3.1-8B-Instruct" | "Llama-3.2-3B-Instruct"
    # ===== End of settings =====
    
    # Prepare training data
    work_config = configs.WORK_CONFIG_MAP[dataset_type]
    ldfh = log_utils.LogDataFrameHelper(work_config.dataset_config.path)
    # Build structured logs
    struct_logs_df = ldfh.build_struct_logs(
        log_path=work_config.dataset_config.path,
        log_format=work_config.dataset_config.fromat,
        start_line=work_config.dataset_config.start_line,
        end_line=work_config.dataset_config.end_line,
    )
    ldfh.save_to_csv(struct_logs_df, f"{work_config.dataset_config.path}-struct-logs.csv")
    # Build semantic logs
    logs_df = ldfh.build_semantic_logs(
        struct_logs_df=struct_logs_df,
        feature_columns=work_config.dataset_config.feat_columns,
        log_regex_replase_fn=log_utils.log_regex_replase,
    )
    ldfh.save_to_csv(logs_df, f"{work_config.dataset_config.path}-semantic-logs.csv")
    # Split training and testing logs
    sampling_fn = samp_utils.train_test_sampling if sampling_type == "our" else samp_utils.logllm_train_test_sampling
    train_logs_df, test_logs_df = sampling_fn(logs_df, environments.TRAIN_RATIO)
    
    # ===== Stage one training =====
    train_unique_logs_df = train_logs_df.drop_duplicates(subset=["log", "label"], inplace=False)
    test_unique_logs_df = test_logs_df.drop_duplicates(subset=["log", "label"], inplace=False)
    # Input and output cases
    train_unique_logs, train_unique_logs_labels = train_unique_logs_df["log"].tolist(), train_unique_logs_df["label"].tolist()
    test_unique_logs, test_unique_logs_labels = test_unique_logs_df["log"].tolist(), test_unique_logs_df["label"].tolist()
    # Train Log Embed Model
    stage_one_train(
        train_case_name=train_case_name, 
        epochs=10, lr=5e-5, safe_batch_size=256,
        train_cases=(train_unique_logs, train_unique_logs_labels),
        test_cases=(test_unique_logs, test_unique_logs_labels),
    )
    
    # ===== Stage two training =====
    log_feature_vector_map = get_log_feature_vector_map(
        train_case_name,
        train_unique_logs + test_unique_logs,
    )
    # Sliding log windows
    build_wins_fn = ldfh.build_count_wins if sliding_window_type == "count" else ldfh.build_time_wins
    train_wins_df = build_wins_fn(df=train_logs_df)
    train_wins_df = ldfh.build_train_adllm_wins(train_wins_df, log_feature_vector_map)
    test_wins_df = build_wins_fn(df=test_logs_df)
    test_wins_df = ldfh.build_train_adllm_wins(test_wins_df, log_feature_vector_map)
    # Input and output cases
    train_wins, train_wins_label = train_utils.parse_train_adllm_wins_df_to_train_case(train_wins_df)
    test_wins, test_wins_label = train_utils.parse_train_adllm_wins_df_to_train_case(test_wins_df)
    # Train Anomaly Detection LLM
    stage_two_train(
        train_case_name=train_case_name,
        epochs=5, lr=5e-5, safe_batch_size=3, top_k_logs=5,
        train_cases=(train_wins, train_wins_label),
        test_cases=(test_wins, test_wins_label),
        base_llm_name=base_llm_name,
        sliding_window_type=sliding_window_type,
        system_name=work_config.system_name,
        log_table_columns=work_config.dataset_config.feat_columns,
    )   
    

if __name__ == "__main__":
    main()