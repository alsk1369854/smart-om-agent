# Smart O&M Agent
**Smart O&M Agent** is a system log anomaly detection framework that combines Large Language Model(LLM) and Retrieval-Augmented Generation(RAG). It can understand complex log semantics and detect high-density log anomalies, and can determine the current system status through the system log window.

## Workflow
<image src="https://raw.githubusercontent.com/alsk1369854/smart-om-agent/refs/heads/master/docs/images/workflow.png" alt="workflow.png">

## Quick Start
### Install
```bash
pip install transformers bitsandbytes peft datasets pandas torch scikit-learn pydantic matplotlib langgraph seaborn
```

### Use
```bash
python use.py
```

### Training
```bash
python train.py
```

## Download banchmarks

### BGL
```bash
# https://zenodo.org/records/8196385/files/BGL.zip?download=1
export DATA_DIR=data
export DATA_NAME=BGL
mkdir -p ${DATA_DIR}/${DATA_NAME} && curl -L https://zenodo.org/records/8196385/files/BGL.zip?download=1 -o ${DATA_DIR}/${DATA_NAME}.zip && unzip ${DATA_DIR}/${DATA_NAME}.zip -d ${DATA_DIR}/${DATA_NAME} && rm ${DATA_DIR}/${DATA_NAME}.zip
```

## Thunderbird
```bash
# https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1
export DATA_DIR=data
export DATA_NAME=Thunderbird
mkdir -p ${DATA_DIR}/${DATA_NAME} && curl -L https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1 -o ${DATA_DIR}/${DATA_NAME}.tar.gz && tar -xzvf ${DATA_DIR}/${DATA_NAME}.tar.gz -C ${DATA_DIR}/${DATA_NAME} && rm ${DATA_DIR}/${DATA_NAME}.tar.gz
```

### Liberty
```bash
# http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/liberty2.gz
export DATA_DIR=data
export DATA_NAME=Liberty
mkdir -p ${DATA_DIR}/${DATA_NAME} && curl -L http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/liberty2.gz -o ${DATA_DIR}/${DATA_NAME}.gz && gunzip -c ${DATA_DIR}/${DATA_NAME}.gz > ${DATA_DIR}/${DATA_NAME}/${DATA_NAME}.log && rm ${DATA_DIR}/${DATA_NAME}.gz
```

## Download based LLMs
```bash
pip install -U "huggingface_hub[cli]"
pip install huggingface_hub[hf_xet]

export HF_TOKEN=<your-huggingface-token>
huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential
```

### BERT
```bash
export SAVE_PATH=hf_models/bert-base-uncased
export MODEL_NAME=google-bert/bert-base-uncased
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

### Gemma2-9B
```bash
export SAVE_PATH=hf_models/gemma-2-9b
export MODEL_NAME=google/gemma-2-9b
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

### Gemma3-4B
```bash
export SAVE_PATH=hf_models/gemma-3-4b-it
export MODEL_NAME=google/gemma-3-4b-it
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

### Llama3.1-8B
```bash
export SAVE_PATH=hf_models/Llama-3.1-8B-Instruct
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

### Llama3.2-3B
```bash
export SAVE_PATH=hf_models/Llama-3.2-3B-Instruct
export MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```