# Smart O&M Agent: An Anomaly Detection Architecture for System Operation and Maintenance Based on RAG
**Smart O&M Agent** is a system log anomaly detection framework that combines Large Language Model(LLM) and Retrieval-Augmented Generation(RAG). It can understand complex log semantics and detect high-density log anomalies, and can determine the current system status through the system log window.

## Workflow
<image src="https://raw.githubusercontent.com/alsk1369854/smart-om-agent/refs/heads/master/docs/images/workflow.png" alt="workflow.png">

## Experiment Results
Experimental Results on BGL, Liberty, and Thunderbird datasets. The best results are indicated using bold typeface.

|       Methods        |    BGL    |    BGL    |    BGL    |  Liberty  |  Liberty  |  Liberty  | Thunderbird | Thunderbird | Thunderbird |             |
| :------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :---------: | :---------: | :---------: | :---------: |
|                      | **Prec.** | **Rec.**  |  **F1**   | **Prec.** | **Rec.**  |  **F1**   |  **Prec.**  |  **Rec.**   |   **F1**    | **Avg. F1** |
|       DeepLog        |   0.166   |   0.988   |   0.285   |   0.751   |   0.855   |   0.800   |    0.017    |    0.966    |    0.033    |    0.373    |
|      LogAnomaly      |   0.176   |   0.985   |   0.299   |   0.684   |   0.876   |   0.768   |    0.025    |    0.966    |    0.050    |    0.372    |
|        PLELog        |   0.595   |   0.880   |   0.710   |   0.795   |   0.874   |   0.832   |    0.808    |    0.724    |    0.764    |    0.769    |
|      FastLogAD       |   0.167   | **1.000** |   0.287   |   0.151   | **0.999** |   0.263   |    0.008    |    0.931    |    0.017    |    0.189    |
|       LogBERT        |   0.165   |   0.989   |   0.283   |   0.902   |   0.633   |   0.744   |    0.022    |    0.172    |    0.039    |    0.355    |
|      LogRobust       |   0.696   |   0.968   |   0.810   |   0.695   |   0.979   |   0.813   |    0.318    |  **1.000**  |    0.482    |    0.702    |
|         CNN          |   0.698   |   0.965   |   0.810   |   0.580   |   0.914   |   0.709   |    0.870    |    0.690    |    0.769    |    0.763    |
|      NeuralLog       |   0.792   |   0.884   |   0.835   |   0.875   |   0.926   |   0.900   |    0.794    |    0.931    |    0.857    |    0.864    |
|        RAPID         |   0.874   |   0.399   |   0.548   |   0.911   |   0.611   |   0.732   |    0.200    |    0.207    |    0.203    |    0.494    |
|        LogLLM        |   0.861   |   0.979   |   0.916   | **0.992** |   0.926   |   0.958   |    0.966    |    0.966    |    0.966    |    0.947    |
| Smart O&M Agent(our) | **0.981** |   0.989   | **0.985** |   0.983   |   0.958   | **0.970** |  **1.000**  |  **1.000**  |  **1.000**  |  **0.985**  |


## Benchmarks 
Benchmark datasets used in the experiments.

|  Datasets   |  \# Logs   | Training Data | Training Data | Training Data | Testing Data | Testing Data | Testing Data |
| :---------: | :--------: | :-----------: | :-----------: | :-----------: | :----------: | :----------: | :----------: |
|             |            |    \# Logs    |    Normal     |   Abnormal    |   \# Logs    |    Normal    |   Abnormal   |
|     BGL     | 4,747,963  |   3,798,387   |   3,519,603   |    278,784    |   949,576    |   879,900    |    69,676    |
|   Liberty   | 5,000,000  |   4,000,007   |   2,719,580   |   1,280,427   |   999,993    |   679,895    |   320,098    |
| Thunderbird | 10,000,000 |   8,000,003   |   7,996,051   |     3,952     |  1,999,997   |  1,999,012   |     985      |


## Using Our Code
### 1. Setup
- Python: 3.12.9
- CUDA: 12
- [Download benchmarks](#download-benchmarks-cli)
- [Download based LLMs](#download-based-llms-cli)

### 2. Install dependencies
```bash
pip install transformers bitsandbytes peft datasets pandas torch scikit-learn pydantic matplotlib langgraph seaborn
```

### 3. Two-stage training of **Smart O&M Agent**
The training results will be stored in `./output/`

```bash
python train.py
```

### 4. Used for system anomaly detection
```bash
python use.py
```


<h1 id="download-benchmarks-cli">
    Download benchmarks Cli
</h1>

## [BGL](https://zenodo.org/records/8196385/files/BGL.zip?download=1)
```bash
export DATA_DIR=data
export DATA_NAME=BGL
mkdir -p ${DATA_DIR}/${DATA_NAME} && curl -L https://zenodo.org/records/8196385/files/BGL.zip?download=1 -o ${DATA_DIR}/${DATA_NAME}.zip && unzip ${DATA_DIR}/${DATA_NAME}.zip -d ${DATA_DIR}/${DATA_NAME} && rm ${DATA_DIR}/${DATA_NAME}.zip
```

## [Thunderbird](https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1)
```bash
export DATA_DIR=data
export DATA_NAME=Thunderbird
mkdir -p ${DATA_DIR}/${DATA_NAME} && curl -L https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1 -o ${DATA_DIR}/${DATA_NAME}.tar.gz && tar -xzvf ${DATA_DIR}/${DATA_NAME}.tar.gz -C ${DATA_DIR}/${DATA_NAME} && rm ${DATA_DIR}/${DATA_NAME}.tar.gz
```

## [Liberty](http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/liberty2.gz)
```bash
export DATA_DIR=data
export DATA_NAME=Liberty
mkdir -p ${DATA_DIR}/${DATA_NAME} && curl -L http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/liberty2.gz -o ${DATA_DIR}/${DATA_NAME}.gz && gunzip -c ${DATA_DIR}/${DATA_NAME}.gz > ${DATA_DIR}/${DATA_NAME}/${DATA_NAME}.log && rm ${DATA_DIR}/${DATA_NAME}.gz
```

<h1 id="download-based-llms-cli">
    Download based LLMs cli
</h1>

## 1. Login Huggingface
```bash
pip install -U "huggingface_hub[cli]"
pip install huggingface_hub[hf_xet]

export HF_TOKEN=<your-huggingface-token>
huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential
```

## [BERT](https://huggingface.co/google-bert/bert-base-uncased)
```bash
export SAVE_PATH=hf_models/bert-base-uncased
export MODEL_NAME=google-bert/bert-base-uncased
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

## [Gemma2-9B](https://huggingface.co/google/gemma-2-9b)
```bash
export SAVE_PATH=hf_models/gemma-2-9b
export MODEL_NAME=google/gemma-2-9b
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

## [Gemma3-4B-IT](https://huggingface.co/google/gemma-3-4b-it)
```bash
export SAVE_PATH=hf_models/gemma-3-4b-it
export MODEL_NAME=google/gemma-3-4b-it
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

## [Llama3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
```bash
export SAVE_PATH=hf_models/Llama-3.2-3B-Instruct
export MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

## [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
```bash
export SAVE_PATH=hf_models/Llama-3.1-8B-Instruct
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
nohup bash -c "huggingface-cli download ${MODEL_NAME} --local-dir ${SAVE_PATH}" &
```

