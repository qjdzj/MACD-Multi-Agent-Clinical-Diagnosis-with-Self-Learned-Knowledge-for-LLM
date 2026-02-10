# MACD: Medical Agent Clinical Decision-making Dataset

The Medical Agent Clinical Decision-making Dataset (MACD) is a comprehensive dataset and evaluation framework for assessing the clinical decision-making capabilities of Large Language Models (LLMs) in medical diagnosis scenarios.

## Project Structure

```
MACD/
├── agents/                    # Agent implementations and prompt templates
├── collaboration_scripts/     # Scripts for analyzing collaborative results
├── configs/                   # Configuration files for experiments
├── dataset/                   # Dataset files and utilities
├── evaluators/                # Evaluation modules for different pathologies
├── icd/                       # ICD code mappings
├── missing_ids/               # Debug utilities for missing patient IDs
├── models/                    # Model interfaces and implementations
├── results/                   # Experiment results storage
├── scripts/                   # Automated experiment execution scripts
├── tests/                     # Unit tests for various components
├── tools/                     # Supporting tools and utilities
└── utils/                     # General utility functions
```

## Getting Started

### Models
- meta-llama/Llama-3.1-8B-Instruct: [https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- meta-llama/Llama-3.1-70B-Instruct: [https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- deepseek-ai/DeepSeek-R1-Distill-Llama-70B: [https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- BioBert: [https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/tree/main](https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/tree/main)


### Installation

```bash
# Clone the repository
git clone https://github.com/qjdzj/MACD-Multi-Agent-Clinical-Diagnosis-with-Self-Learned-Knowledge-for-LLM.git
cd MACD

# Install dependencies
conda create -n macd python=3.10
pip install -r requirements.txt
```

## Usage

### Data

You can obtain the complete dataset from the [data/](./MACD-data/) directory, the actual data used for testing from [test_set/](./MACD-data/test_set/) directory, and the data used for human physicians evaluation can be obtained from [MACD-human/](./MACD-data/MACD-human/)


### Specialized Prompts

Different agent prompts can be imported from the [agents/](./agents/) directory:
- Self-learned knowledge
- Mayo Clinic guidelines
- Professional guidelines

### Inference Script

The main inference script is [infer.py](./infer.py), which supports various configurations through command-line arguments:

```bash
python infer.py \
    model="Llama3Instruct8B" \
    pathology="pneumonia" \
    [other_options]
```

Key options include:
- `model`: Specifies the LLM to use (Llama3Instruct8B, Llama3Instruct70B, Deepseek-Llama70B-distill)
- `pathology`: Target disease for diagnosis (appendicitis, cholecystitis, diverticulitis, pancreatitis, pneumonia, pulmonary embolism, pericarditis)
- `criteria`: Enable diagnostic criteria
- `guideline`: Enable clinical guidelines
- `fewshot`: Enable few-shot learning examples

### Running Experiments

The project provides automated experiment execution through bash scripts located in the ./scripts/ directory. These scripts call the main inference script infer.py with different configurations.

To run experiments, simply execute the desired bash script:

```bash
# Run baseline experiments
bash scripts/baseline.sh

# Run chain-of-thought experiments
bash scripts/cot.sh

# Run few-shot experiments
bash scripts/fewshot.sh

# Run experiments with professional guidelines
bash scripts/guideline_professional.sh

# Run experiments with Mayo Clinic guidelines
bash scripts/guideline_mayo.sh

# Run human cases experiments
bash scripts/human_cases.sh

# Run inconsistency analysis with past diagnosis
bash scripts/inconsistent_with_past.sh
```


### Evaluation

Results can be evaluated using the [evaluate.py](./evaluate.py) script:

```bash
bash evaluate.sh
```

The [collaboration_scripts/](./collaboration_scripts/) directory contains analysis tools for evaluating MACD-human collaboration workflow experiments.

### Self learned knowledge acquisition

You can run the model from the script in the [self-knowledge_scripts/](./self-knowledge_scripts/) directory to obtain the corresponding self learned knowledge and fully experience the entire process of MACD. However, this will be a long process

## Acknowledgements

This project is built upon and extends the work from the MIMIC Clinical Decision Making Framework by Hager, P. et al. We gratefully acknowledge the significant contributions of the original authors whose foundational work in clinical evaluation of large language models has enabled our research extensions.

## Citation

If you use this code or dataset in your research, please cite our paper:

```
@article{li2025macd,
  title={MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM},
  author={Li, Wenliang and Yan, Rui and Zhang, Xu and Chen, Li and Zhu, Hongji and Zhao, Jing and Li, Junjun and Li, Mengru and Cao, Wei and Jiang, Zihang and others},
  journal={arXiv preprint arXiv:2509.20067},
  year={2025}
}
```