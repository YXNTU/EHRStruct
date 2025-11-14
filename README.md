
# EHRStruct: A Comprehensive Benchmark Framework for Evaluating Large Language Models on Structured Electronic Health Record Tasks

EHRStruct is a comprehensive benchmark for evaluating large language models  on structured electronic health record (EHR) tasks.
It defines 11 clinically grounded tasks across 6 categories, provides 2,200 standardized evaluation samples derived from Synthea and eICU datasets, and enables systematic comparison across both general-purpose and medical-domain LLMs.

## 🌈 Update

- [2025.11.08] Accepted as an **Oral presentation** at AAAI 2026. 🎉


## 📄 Paper Versions

- AAAI 2026: [Paper (coming soon)]()  
- [arXiv](https://arxiv.org/abs/2511.08206)


## 🧭 Framework Overview

<p align="center">
  <img src="source/Overview of EHRStruct.jpg" width="800">
</p>

**Overview of EHRStruct.**  
The figure illustrates the four key components of the benchmark:  
(1) task synthesis through clinical needs induction and task distillation from prior research;  
(2) taxonomy construction based on clinical scenarios and reasoning levels;  
(3) task-specific sample extraction from real and synthetic EHR data;   
(4) the model evaluation pipeline, including table input, format conversion, model inference, and answer evaluation.  

## 🧪 Experiments

### Dataset

We use two datasets in this benchmark: Synthea and eICU.  
The Synthea dataset originates from the open-source synthetic patient generator [Synthea](https://github.com/synthetichealth/synthea). It contains fully simulated patient records and does not include any identifiable or real-world information.  

The preprocessed data 👉  [Google Drive](https://drive.google.com/drive/folders/1-XXajeBbjDJxsX1KZ6MnxRP_qwHoAylS?usp=drive_link)


After downloading, unzip the files into the `EHRStruct/` directory to run experiments directly. 

🔴 **Note:** The released data have been reorganized and standardized, so numerical differences may appear but do not affect the overall conclusions.


### Code Structure

The repository consists of five main folders:

```
EHRStruct/
│
├── data/
│   ├── aggregation/        # sample_001.csv–sample_100.csv; query_answer_D-R1.csv, D-R2.csv, D-R3.csv
│   ├── arithmetic/         # sample_001.csv–sample_100.csv; query_answer_D-R4.csv, D-R5.csv
│   ├── death/              # sample_001.csv–sample_100.csv; query_answer_K-R1.csv
│   ├── disorder/           # sample_001.csv–sample_100.csv; query_answer_K-R2.csv
│   ├── filter/             # sample_001.csv–sample_100.csv; query_answer_D-U1.csv, D-U2.csv
│   ├── medications/        # sample_001.csv–sample_100.csv; query_answer_K-R3.csv
│   └── snomed/             # sample_001.csv–sample_100.csv; query_answer_K-U1.csv
│
├── EHRMaster/              # EHRMaster implementation
│   ├── __init__.py
│   ├── LLMCaller.py
│   └── run.py
│
├── Gemini/                 # Gemini API interface (Google DeepMind models)
│   ├── __init__.py
│   └── LLMCaller.py
│
├── Openai/                 # GPT API interface (OpenAI models)
│   ├── __init__.py
│   └── LLMCaller.py
│
└── Siliconflow/            # Qwen / DeepSeek API interface (main execution example)
    ├── __init__.py
    ├── LLMCaller.py
    └── run.py
```


### Environment

- Python ≥ 3.9  
- PyTorch ≥ 2.6  
- Transformers ≥ 4.51  

Here we provide a configuration file to install the extra requirements (if needed):

```bash
conda install --file requirements.txt
```

> 🔴 **Note:** This file will not install `torch` or `torchvision`. Please install them separately according to the CUDA version of your graphics card via [PyTorch](https://pytorch.org/).


### Evaluation

To evaluate models, enter the `Siliconflow` directory and run:

```bash
cd Siliconflow
python run.py --llm Qwen72B --task aggregation --type txt --k 0

# Default options:
# --llm    [Qwen7B, Qwen14B, Qwen32B, Qwen72B, deepseekV2.5, deepseekV3]
# --task   [filter (D-U1/U2), aggregation (D-R1/R2/R3), arithmetic (D-R4/R5),
#           snomed (K-U1), death (K-R1), disorder (K-R2), medications (K-R3)]
# --type   [txt (plain text conversion), latex (special character separation),
#           hyper (graph-structured representation), sgen (natural language description)]
# --k      [Number of few-shot examples, 0 for zero-shot]
# Results will be saved in Siliconflow/output/
```

> 🔴 **Note:** For evaluating Gemini and GPT models, please refer to the corresponding API call examples in `Gemini/LLMCaller.py` and `Openai/LLMCaller.py`.


### EHRMaster Evaluation

Our work shows that EHRMaster performs particularly well on **Data-Driven tasks**. We therefore release the evaluation setup for these tasks.

```bash
cd EHRMaster
python run.py --llm Qwen72B --task D-U1

# Default options:
# --llm    [Qwen7B, Qwen14B, Qwen32B, Qwen72B]
# --task   [D-U1, D-U2, D-R1, D-R2, D-R3, D-R4, D-R5]
# Results will be saved in EHRMaster/output/
```

### Citation

If you find EHRStruct helpful in your research, please cite our paper:

```bibtex
@article{yang2025ehrstruct,
  title={EHRStruct: A Comprehensive Benchmark Framework for Evaluating Large Language Models on Structured Electronic Health Record Tasks},
  author={Yang, Xiao and Zhao, Xuejiao and Shen, Zhiqi},
  journal={arXiv preprint arXiv:2511.08206},
  year={2025}
}
```

### License
EHRStruct is released under the MIT License. Our codes must only be used for the purpose of research.



