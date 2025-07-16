import os
import torch
import numpy as np
from torch import nn
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import Self, Any, Literal
from . import types, prompts

class LogEmbedModel(nn.Module):
    MAX_TOKEN_LEN = 256

    bert: BertModel
    bert_tokenizer: BertTokenizerFast
    device: torch.device

    @classmethod
    def from_pretrained(cls, *, path: str, device: torch.device | str = "auto") -> Self:
        save_paths = cls._get_save_paths(path)
        model = cls()
        # bert
        model.bert_tokenizer = BertTokenizerFast.from_pretrained(save_paths.bert)
        model.bert = BertModel.from_pretrained(save_paths.bert, device_map=device)
        model.device = model.bert.device
        # classifier
        classifier_state_dict = torch.load(save_paths.classifier, map_location=model.device)
        model.classifier.load_state_dict(classifier_state_dict)
        model.classifier.to(model.device)
        return model
    
    @classmethod
    def new(cls, *, bert_path: str, device: torch.device | str = "auto") -> Self:
        model = cls()
        model.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_path)
        model.bert = BertModel.from_pretrained(bert_path, device_map=device)
        model.device = model.bert.device
        model.classifier.to(model.device)
        return model
        
    @classmethod 
    def _get_save_paths(cls, save_base: str) -> types.LogEmbedModelSavePaths:
        return types.LogEmbedModelSavePaths(
            bert=os.path.join(save_base, "bert"),
            classifier=os.path.join(save_base, "classifier.pt"),
        )

    def __init__(self):
        super().__init__()
        in_features = 768  # BERT 隱藏層維度
        # 定義分類器（全連接層）
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features // 2, 1),
        )
        
    def forward(self, logs: list[str]):
        tokenized = self.bert_tokenizer(
            logs, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.MAX_TOKEN_LEN
        ).to(self.device)
        # BERT 模型輸出，取 [CLS] 向量（batch_size, in_features）
        outputs = self.bert(**tokenized).last_hidden_state[:, 0] 
        # 經過全連接層進行分類
        return self.classifier(outputs).squeeze(-1) 
    
    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        save_paths = self._get_save_paths(path)
        self.bert.save_pretrained(save_paths.bert)
        self.bert_tokenizer.save_pretrained(save_paths.bert)
        torch.save(self.classifier.state_dict(), save_paths.classifier)

    def pred(self, logs: list[str]):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(logs)
            return torch.sigmoid(outputs).cpu().numpy()



class AnomalyDetectionLLM(nn.Module):
    MAX_TOKEN_LEN = 512

    llm: PeftModel
    llm_tokenizer: AutoTokenizer
    device: torch.device
    max_logs_tokens_len: int
    system_prefix_tokens: Any
    system_suffix_tokens: Any
    abnormal_output_tokens: Any
    normal_output_tokens: Any

    @classmethod
    def from_pretrained(
        cls, 
        *, 
        save_path: str, 
        base_llm_path: str,  
        device: str | torch.device = "auto", 
        system_name: str | None = None, 
        field_names: str = "",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> Self:
        os.makedirs(save_path, exist_ok=True)
        save_paths = cls._get_save_paths(save_path)
        model = cls()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # load the model into memory using 4-bit precision
            bnb_4bit_use_double_quant=False,  # use double quantition
            bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
            bnb_4bit_compute_dtype=torch_dtype  # use hf for computing when we need
        )
        base_llm = AutoModelForCausalLM.from_pretrained(
            base_llm_path, 
            torch_dtype=torch_dtype, 
            quantization_config=bnb_config, 
            device_map=device,
            attn_implementation="eager",
        )
        model.llm = PeftModel.from_pretrained(base_llm, save_paths.llm, is_trainable=True)
        model.llm_tokenizer = AutoTokenizer.from_pretrained(save_paths.llm, padding_side="right")
        model.llm_tokenizer.pad_token_id = model.llm_tokenizer.eos_token_id if model.llm_tokenizer.pad_token_id is None else model.llm_tokenizer.pad_token_id
        model.device = model.llm.device
        model.set_system_name_and_field_names(system_name=system_name, field_names=field_names)
        return model
    
    @classmethod
    def new(
        cls, 
        *, 
        base_llm_path: str, 
        device: str | torch.device = "auto", 
        system_name: str | None = None, 
        field_names: str = "",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> Self:
        model = cls()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # load the model into memory using 4-bit precision
            bnb_4bit_use_double_quant=False,  # use double quantition
            bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
            bnb_4bit_compute_dtype=torch_dtype  # use hf for computing when we need
        )
        base_llm = AutoModelForCausalLM.from_pretrained(
            base_llm_path, 
            torch_dtype=torch_dtype, 
            quantization_config=bnb_config, 
            device_map=device,
            attn_implementation="eager",
        )
        print(base_llm.config._attn_implementation)
        model.llm = get_peft_model(
            base_llm,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
        )
        model.llm_tokenizer = AutoTokenizer.from_pretrained(base_llm_path, trust_remote_code=True, padding_side="right")
        model.llm_tokenizer.pad_token_id = model.llm_tokenizer.eos_token_id if model.llm_tokenizer.pad_token_id is None else model.llm_tokenizer.pad_token_id
        model.device = model.llm.device
        model.set_system_name_and_field_names(system_name=system_name, field_names=field_names)
        return model

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _get_save_paths(self, save_base: str) -> types.AnomalyDetecteLLMSavePaths:
        return types.AnomalyDetecteLLMSavePaths(
            llm=os.path.join(save_base, "llm"),
        )

    def save_pretrained(self, *, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        save_paths = self._get_save_paths(path)
        self.llm.save_pretrained(save_paths.llm)
        self.llm_tokenizer.save_pretrained(save_paths.llm)

    def set_system_name_and_field_names(self, *, system_name: str | None, field_names: str) -> None:
        prefix_prompt = prompts.ANOMALY_DETECTE_LLM_PREFIX_INSTRUCTION_TEMPLATE.format(
            system_name="" if system_name is None else f" {system_name} ",
            field_names=field_names
        )
        prefix_tokens = self.llm_tokenizer(
            prefix_prompt,  
            return_tensors="pt"
        )["input_ids"][0]
        shffix_tokens = self.llm_tokenizer(
            prompts.ANOMALY_DETECTE_LLM_SUFFIX_INSTRUCTION, 
            add_special_tokens=False, # Omit [BOS] token
            return_tensors="pt"
        )["input_ids"][0]
        abnormal_output_tokens = self.llm_tokenizer("Abnormal", add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        normal_output_tokens = self.llm_tokenizer("Normal", add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        instruction_len = len(prefix_tokens) + len(shffix_tokens)
        output_len = max(len(abnormal_output_tokens), len(normal_output_tokens)) + 1 # eos token

        self.max_logs_tokens_len = self.MAX_TOKEN_LEN - (instruction_len + output_len)
        if self.max_logs_tokens_len <= 16: 
            raise Exception("need more tokens length to contain the logs tokens.")
        
        self.system_prefix_tokens = prefix_tokens
        self.system_suffix_tokens = shffix_tokens
        self.abnormal_output_tokens = abnormal_output_tokens
        self.normal_output_tokens = normal_output_tokens

    # def unfreeze_llm_params(self) -> None:
    #     for name, param in self.llm.named_parameters():
    #         if 'lora' in name:
    #             param.requires_grad = True

    def _get_log_win_tokens(self, log_win: list[str]) -> torch.Tensor:
        log_tokens_list = []
        total_log_tokens_len = 0
        index = 0
        while (index < len(log_win)) and (total_log_tokens_len < self.max_logs_tokens_len):
            log_tokens = self.llm_tokenizer(f"{log_win[index]}\n", add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            can_included = (len(log_tokens) + total_log_tokens_len) <= self.max_logs_tokens_len
            if not can_included:
                log_tokens = log_tokens[:self.max_logs_tokens_len - total_log_tokens_len]
            total_log_tokens_len += len(log_tokens)
            log_tokens_list.append(log_tokens)
            index += 1

        return torch.cat(log_tokens_list, dim=0)

    def forward(
        self, 
        log_wins: list[list[str]], 
        targets: list[int],
    ):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for log_win, target in zip(log_wins, targets):
            log_win_tokens = self._get_log_win_tokens(log_win)
            output_tokens = self.abnormal_output_tokens if target == 1 else self.normal_output_tokens
            
            # 建立 input_ids
            eos_token = self.llm_tokenizer.eos_token_id
            input_ids = torch.cat([
                self.system_prefix_tokens.to(torch.device("cpu")), 
                log_win_tokens.to(torch.device("cpu")), 
                self.system_suffix_tokens.to(torch.device("cpu")), 
                output_tokens.to(torch.device("cpu")),
                torch.tensor([eos_token], dtype=torch.long, device=torch.device("cpu")),
            ], dim=0)

            # 填充至符合 MAX_TOKEN_LEN
            pad_len = self.MAX_TOKEN_LEN - len(input_ids)
            pad_token = self.llm_tokenizer.pad_token_id
            if pad_len > 0:
                input_ids = torch.cat([input_ids, pad_token * torch.ones(pad_len, dtype=torch.long, device=torch.device("cpu"))])

            # 建立 attention mask
            attention_mask = (input_ids != pad_token).long().to(torch.device("cpu"))
            
            # 建立 label 只計算 ouput_tokens 的 loss
            input_tokens_len = len(self.system_prefix_tokens) + len(log_win_tokens) + len(self.system_suffix_tokens)
            output_tokens_len = len(output_tokens) + 1 # eos token
            labels = torch.full_like(input_ids, -100)
            labels[input_tokens_len:input_tokens_len + output_tokens_len] = input_ids[input_tokens_len:input_tokens_len + output_tokens_len]

            # 添加到任務列表中
            input_ids_list.append(input_ids.unsqueeze(0))
            attention_mask_list.append(attention_mask.unsqueeze(0))
            labels_list.append(labels.unsqueeze(0))

        # batchify
        input_ids = torch.cat(input_ids_list, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask_list, dim=0).to(self.device)
        labels = torch.cat(labels_list, dim=0).to(self.device)

        return self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, log_win: list[str]) -> str:
        self.eval()
        with torch.no_grad():
            log_win_tokens = self._get_log_win_tokens(log_win)
            # 建立 input_ids 與 attention_mask
            input_ids = torch.cat([self.system_prefix_tokens, log_win_tokens, self.system_suffix_tokens], dim=0).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            input_len = len(input_ids[0])

            generation = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max(len(self.abnormal_output_tokens), len(self.normal_output_tokens)) + 10, 
                pad_token_id=self.llm_tokenizer.pad_token_id,
                do_sample=False, 
                temperature=None,
                top_k=None,
                top_p=None, 
            )
            generation = generation[0][input_len:]

            return self.llm_tokenizer.decode(generation, skip_special_tokens=True).strip()

