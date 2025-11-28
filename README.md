
<p align="center">
  <img src="https://raw.githubusercontent.com/YXNTU/EHRStruct/main/source/logo_horizontal.jpg" width="700">
</p>


<a href='https://arxiv.org/abs/2511.08206'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://yxntu.github.io/proEHRStruct/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

💻 This repository includes all data of the **EHRStruct** benchmark and the official implementation of the proposed model **EHRMaster**.

✅ The paper [**EHRStruct: A Comprehensive Benchmark Framework for Evaluating Large Language Models on Structured Electronic Health Record Tasks**](https://arxiv.org/abs/2511.08206) has been accepted by [The 40th AAAI Conference on Artificial Intelligence (AAAI) 2026](https://aaai.org/conference/aaai/aaai-26/).

**EHRStruct** is a comprehensive benchmark for evaluating large language models on structured electronic health record (EHR) tasks.
It defines **11 clinically grounded tasks** across 6 categories, provides **2,200 standardized evaluation samples** derived from Synthea and eICU datasets, and enables systematic comparison across both general-purpose and medical-domain LLMs.

#### 📄 Paper Versions
- AAAI 2026 (OTW)
- [arXiv](https://arxiv.org/abs/2511.08206)

#### Authors
[Xiao Yang](https://yxntu.github.io/),  [Xuejiao Zhao*](https://zxjwudi.github.io/xuejiaozhao/), [Zhiqi Shen](https://scholar.google.com.sg/citations?user=EA2T_lwAAAAJ&hl=en)

**Nanyang Technological University  &nbsp; | &nbsp; LILY Research Centre (NTU) &nbsp; |&nbsp; ANGEL Research Institute (NTU)**

\* Corresponding author

[![Stargazers repo roster for @YXNTU/EHRStruct](https://reporoster.com/stars/YXNTU/EHRStruct)](https://github.com/YXNTU/EHRStruct/stargazers)

---

## 🌈 News
* **[2025.11.16]** We release the [ProjectPage](https://yxntu.github.io/proEHRStruct/) of **EHRStruct**.
* **[2025.11.16]** We release the preprint of **EHRStruct** on [arXiv](https://arxiv.org/abs/2511.08206).
* **[2025.11.13]** We release github repository of **EHRStruct** and **EHRMaster**. 💪 Come to take the challenge！
* **[2025.11.08]** Accepted as an **Oral presentation** to AAAI 2026. 🎉

## 🧭 Framework Overview

<p align="center">
  <img src="source/Overview of EHRStruct.jpg" width="800">
</p>
    <p align="center"><em>Figure 1: Overview of EHRStruct.</em></p >

The figure illustrates the four key components of the benchmark:  
 **(1)** task synthesis through clinical needs induction and task distillation from prior research;  
 **(2)** taxonomy construction based on clinical scenarios and reasoning levels;  
 **(3)** task-specific sample extraction from real and synthetic EHR data;   
 **(4)** the model evaluation pipeline, including table input, format conversion, model inference, and answer evaluation.  

## 🧪 Experiments

### Dataset

We use two datasets in this benchmark: **Synthea** and **eICU**.  
- **Synthea** dataset originates from the open-source synthetic patient generator [Synthea](https://github.com/synthetichealth/synthea). It contains fully simulated patient records and does not include any identifiable or real-world information.
  Users can either generate their own data or directly download the preprocessed data from 👉 [Google Drive](https://drive.google.com/drive/folders/1-XXajeBbjDJxsX1KZ6MnxRP_qwHoAylS?usp=drive_link).
  After downloading, unzip the files into the `EHRStruct/` directory to run experiments directly.

- **eICU** dataset originates from the [eICU Collaborative Research Database](https://physionet.org/content/eicu-crd/2.0/).
  Users must obtain **credentialed access** via PhysioNet to download the raw data. We provide the preprocessing code of this dataset in the [`eICU/`](https://github.com/YXNTU/EHRStruct/tree/main/eICU) directory of this project.

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

## Citation

If you find EHRStruct helpful in your research, please cite our paper:

```bibtex
@article{yang2025ehrstruct,
  title={EHRStruct: A Comprehensive Benchmark Framework for Evaluating Large Language Models on Structured Electronic Health Record Tasks},
  author={Yang, Xiao and Zhao, Xuejiao and Shen, Zhiqi},
  journal={arXiv preprint arXiv:2511.08206},
  year={2025}
}
```

## License
EHRStruct is released under the MIT License. Our codes must only be used for the purpose of research.



