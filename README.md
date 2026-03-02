# 7043SCN — Task 1: Cybersecurity Incident Classification (Phi-3-mini + QLoRA)

**Name:** Sai Sumanth Desagani  
**Student ID:** 16464285  
**Task:** 1 (Cybersecurity Incident Classification)  
**Model:** Phi-3-mini-4k-instruct (4-bit) + QLoRA (PEFT) + LoRA Adapter Pruning

---

## Overview

This project builds a **23-class cybersecurity incident classifier** by fine-tuning **Phi-3-mini-4k-instruct** using **QLoRA (4-bit PEFT)** under consumer GPU constraints (**RTX 3050 6GB**).  
The classifier is implemented as a **constrained generation task**: the model must output **only a single numeric class ID (1–23)** for each incident scenario.

The project evaluates:
1) **Baseline** (pretrained model, zero fine-tuning)  
2) **Full Fine-Tuning feasibility** (infeasible on 6GB VRAM)  
3) **QLoRA fine-tuning** (prompt-masked supervision)  
4) **Model optimisation:** **20% magnitude pruning** on LoRA adapters (zero degradation)

---

## Dataset

- **Source:** `oceancharcoal/Cybersecurity_attack_dataset` (Hugging Face)
- **Total rows:** 14,133
- **Used:** 12,500 (stratified subsample)
- **Split:** Stratified **80/10/10**
  - Train: 10,000
  - Validation: 1,250
  - Test: 1,250
- **Eval subset:** fixed 300-sample subset from test (consistent across all stages)
- **Labels:** mapped from raw categories into **23 clean classes**

---

## Results (Fixed 300-sample Eval)

- **Baseline (pretrained, no fine-tuning):**  
  Accuracy **0.2067**, Macro-F1 **0.1773**
- **QLoRA (PEFT):**  
  Accuracy **0.7567**, Macro-F1 **0.7596**
- **QLoRA + Pruning (20%):**  
  Accuracy **0.7567**, Macro-F1 **0.7596** (**zero degradation**)

Hardware footprint:
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB)
- **Peak VRAM (QLoRA):** ~4.66 GB
- **QLoRA training time:** ~420.9 minutes (~7 hours)

---

## Repository Structure
task1_cyber_incident_classification/
notebooks/
task1_pipeline.ipynb # baseline → QLoRA → pruning (main notebook)
results_figures/
fig1_accuracy_f1_by_stage.png
fig2_perclass_f1.png
fig3_precision_recall_scatter.png
fig4_progression_curve.png
fig5_hardware_utilisation.png
qlora_out/ # saved QLoRA adapter checkpoint
qlora_pruned_lora/ # saved pruned adapter checkpoint (20%)
report/
task1_report.pdf # final two-column IEEE-style report
requirements.txt # python dependencies
README.md # this file


---

## How to Run

### 1) Install dependencies
Create an environment and install packages:

```bash
pip install -r requirements.txt
