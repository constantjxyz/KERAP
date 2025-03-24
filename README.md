
# KERAP

This is the code repository for paper: KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs and EHR, which aims to enhance LLM's zero-shot diagnosis prediction task utilizing KG through multi-agent framework and multi-stage reasoning process. 

## Task Description

Medical diagnosis prediction, which is the task of predicting a patient’s future health risks based on their historically observed medical data such as electronic health records (EHRs), plays a vital role in enabling accurate healthcare and early interventions.

## Method Framework
KERAP consists of three key components: the linkage agent, the retrieval agent, and the prediction agent. The linkage agent maps the predicted disease to a biomedical KG, establishing connections with relevant entities. The retrieval agent then queries the KG to extract and summarize related knowledge, categorizing the results into positive (e.g.,“symptom X indicates condition Y”) and negative knowledge (e.g., “symptom X rules out condition Z”) for inclusion and exclusion criteria. Finally, the prediction agent AP R integrates patient records with the extracted structured
knowledge, leveraging multi-stage reasoning to achieve zero-shot diagnosis prediction.

## Linkage Agent

The **linkage** folder contains code for concept linking and candidate generation using both embedding-based methods and prompt-based models.

- **matching.py**: Core entry point for running the linking process between queries and candidate concepts.
- **gen_response.py**: Generates responses for linking using LLM.
- **gen_candidates/**: 
  - **gen_embedding.py**: Loads embeddings and creates candidate pools for concept matching.
  - **match_embedding.py**: Performs nearest neighbor search between query and candidate embeddings.
  - **utils/**: Utility scripts used in the matching process.
    - **distances.py**: Implements distance functions (e.g., cosine similarity).
    - **others.py**: Helper functions.
- **utils/**:
  - **metrics.py**: Evaluation metrics (e.g., accuracy, precision@k) for the linking task.
  - **prompts.py**: Prompt templates and response patterns for LLM-based linking.
  - **retrieval.py**: Embedding and candidate retrieval utilities.
  - **others.py**: Miscellaneous utilities.

### Usage Example

```bash
python linkage/matching.py
```

Options include:
- `--mode`: choose between `embedding` or `prompt` for matching methods.
- `--top_k`: number of top candidates to return.


## Retrieval Agent

The **retrieval** folder contains scripts for retrieving positive and negative samples using LLM prompting and other filtering strategies.

- **extraction_positive.py**: Extracts positive knowledge for prediction (inclusion criteria).
- **extraction_negative.py**: Identifies negative knowledge for prediction (exclusion criteria) .
- **prompts.py**: Contains prompt templates specific to information retrieval.
- **utils.py**: Shared helper functions for extraction and logging.
- **gen_response.py**: Generates LLM responses to be parsed by the positive/negative extractors.

### Usage Example

```bash
python retrieval/extraction_positive.py
python retrieval/extraction_negative.py
```

You may configure your prompting settings and parsing rules in `prompts.py`.


## Prediction Agent

- **prediction/**: Core module for data preprocessing, model inference, and evaluation.
  - **main.py**: Entry point script for running predictions.
  - **utils/raw_dataset.py**: Utilities for loading and processing raw input datasets.
  - **utils/evaluation.py**: Functions for computing evaluation metrics (e.g., precision, recall, F1-score).

- **README.md**: Project documentation.
- **requirements.txt**: Python package dependencies.

### Usage

Prepare your dataset in the expected format (e.g., CSV or structured dictionary). See `utils/raw_dataset.py` for loading logic. Typically, the input includes patient records with features and associated diagnosis labels.


You can run the prediction pipeline by executing:

```bash
python prediction/main.py
```

Optional arguments (define these in `main.py`):
- `--model`: specify the prediction model (e.g., GPT-4o-mini).

The evaluation metrics such as precision, recall, and F1-score are printed after model inference. You can also modify `utils/evaluation.py` to include additional evaluation metrics (e.g., ROC-AUC).

## Requirements

- Python 3.11.5
- Required libraries listed in `requirements.txt`.

## Data 

We use a large-scale public knowledge graph, iBKH (from https://github.com/wcm-wanglab/iBKH), as the primary KG dataset. As for the patient context information, we use patient-specific data from two EHR datasets: MIMIC-III (from https://physionet.org/content/mimiciii/1.4/) and PROMOTE (private dataset). Due to the sensitive nature of medical data and privacy considerations, there are restrictions on data sharing. To gain access to the two patient-specific datasets, appropriate training and credentials may be required (https://physionet.org/). For further assistance with data access or other related inquiries, please feel free to reach out to our author team.

## Acknowledgements
We would like to thank the authors from PromptLink (https://github.com/constantjxyz/PromptLink), iBKH (https://github.com/wcm-wanglab/iBKH) for their open-source efforts.
