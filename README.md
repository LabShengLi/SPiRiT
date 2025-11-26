# **Inferring single-cell spatial gene expression with tissue morphology via deep learning**

**Authors**: Yue Zhao, Elaheh Alizadeh, Yang Liu, Xu Ming, J Matthew Mahoney*, Sheng Li*


  **Contact**: Matt.Mahoney@jax.org; Sli68423@usc.edu



## Description 
In this work, we developed a framework to use Vision Transformer (ViT) to help map their tissue context, e.g., transcriptomic signatures to imaging signatures. The framework, SPiRiT (Spatial Omics Prediction and Reproducibility integrated Transformer), systematically links spatial single-cell gene expressions with the morphology of the local cellular microenvironment via ViT models. SPiRiT predicts single-cell spatial gene expression from histopathological images of human and whole mouse pup, evaluated by 10x Genomics Xenium datasets. Furthermore, the framework quantifies the confidence level of the High Attention Region (HAR) interpreting the trained ViT by measuring the consistency of attention maps of the same H&E image used in the training step or validation step.

<img width="941" alt="image" src="https://github.com/LabShengLi/SPiRiT/assets/8755378/b4006bc5-68a3-472a-80e8-040e561e295e">


# 🚀 Reproducing SPiRiT Results with Docker

To make SPiRiT easy to use and fully reproducible, we provide a **ready-to-run Docker container** that includes the full environment, dependencies, and scripts used in the manuscript. This allows anyone to recreate all experimental results with minimal setup.

---

## 📦 1. Pull & Launch the Container

```bash
docker run --memory=500g --platform=linux/amd64 -it \
    yuz12012/spirit_container:latest bash
```

> **Note:** We recommend assigning substantial memory (e.g., `--memory=500g`) since SPiRiT processes large histology and gene-expression datasets.

---

## 🧬 2. Activate the Conda Environment

Inside the container:

```bash
conda activate torch_env
```

This activates the SPiRiT runtime environment containing all required packages (PyTorch, OpenCV, NumPy, SciPy, etc.).

---

## 📁 3. Navigate to the Workspace

```bash
cd /workspace
```

This directory contains:

- `Xenium/` – Xenium breast cancer analysis  
- `HER2/` – HER2 spatial transcriptomics analysis  
- Model code and utility functions  
- Output folders for experiment results  

---

# 🧫 Running Experiments

## 🔬 Xenium Breast Cancer Dataset

Run the Xenium experiment:

```bash
python Xenium/SPiRiT_xenium.py
```

All outputs (figures, predictions, and tables) will be saved to:

```
/workspace/output
```

---

## 🧪 HER2 Spatial Transcriptomics Dataset

Run the full HER2 pipeline:

```bash
python HER2/CVPrediction.py
python HER2/WangCorrelation.py
python HER2/Visualization.py
```

HER2 experiment results will be saved to:

```
/workspace/output/spirit_cv
```

---

# 📄 Summary of Commands

| Dataset | Commands | Output Directory |
|--------|----------|------------------|
| **Xenium Breast Cancer** | `python Xenium/SPiRiT_xenium.py` | `/workspace/output` |
| **HER2 ST Dataset** | `python HER2/CVPrediction.py`<br>`python HER2/WangCorrelation.py`<br>`python HER2/Visualization.py` | `/workspace/output/spirit_cv` |








