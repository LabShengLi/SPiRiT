# **Training Attention and Validation Attention Consistency (TAVAC)**

**Authors**: Yue Zhao, Elaheh Alizadeh, Yang Liu, Xu Ming, J Matthew Mahoney*, Sheng Li*


  **Contact**: sheng.li@jax.org


## Description 
In this work, we developed a framework to use Vision Transformer (ViT) to help map their tissue context, e.g., transcriptomic signatures to imaging signatures. The framework, SPiRiT (Spatial Omics Prediction and Reproducibility integrated Transformer), systematically links spatial single-cell gene expressions with the morphology of the local cellular microenvironment via ViT models. SPiRiT predicts single-cell spatial gene expression from histopathological images of human and whole mouse pup, evaluated by 10x Genomics Xenium datasets. Furthermore, the framework quantifies the confidence level of the High Attention Region (HAR) interpreting the trained ViT by measuring the consistency of attention maps of the same H&E image used in the training step or validation step.

## Test

