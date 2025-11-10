
# EHRStruct: A Comprehensive Benchmark Framework for Evaluating Large Language Models on Structured Electronic Health Record Tasks

EHRStruct is a comprehensive benchmark for evaluating large language models  on structured electronic health record (EHR) tasks.
It defines 11 clinically grounded tasks across 6 categories, provides 2,200 standardized evaluation samples derived from Synthea and eICU datasets, and enables systematic comparison across both general-purpose and medical-domain LLMs.

## 🌈 Update

- [2025.11.08] Accepted as an **Oral presentation** at AAAI 2026. 🎉


## 📄 Paper Versions

- AAAI 2026: [Paper (coming soon)]()  
- arXiv: [Paper (coming soon)]()


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









