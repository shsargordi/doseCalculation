# Head & Neck Dose Prediction on CT

This repository provides code and pretrained weights for dose distribution prediction in Head & Neck Cancer (HNC) radiotherapy.

The model was trained and validated on the public OpenKBP Head & Neck dataset, and it can be applied to other HNC datasets (e.g., an in-house cohort) for inference.



## Overview

This repository provides **dose prediction** and **dosimetric evaluation** tools for **Head & Neck (HNC) radiotherapy**.

For each patient, the pipeline can take as input:
- **Real CT (rCT)**
- **Synthetic CT (sCT)**
- **CBCT**
- Corresponding **radiotherapy structures** (**PTVs + OARs**)

and produces:
- **Predicted dose distributions** for the provided image modality/modalities
- **Dosimetric evaluation metrics**, including:
  - **Dose MAE** computed over the **head & neck dose mask**
  - **DVH errors/differences**, reported as:
    - an overall average DVH score aggregated across **PTV70**, **PTV63**, **PTV56** and all included **OARs**, and
    - **per-structure DVH metrics** computed separately for each target volume and each OAR

For additional details on the methodology and evaluation protocol, please refer to the paper included in this repository:  
`/....../doseCalculation/article.pdf`




# Pretrained Weights

Pretrained weights are provided with this codebase.

Path: `/weights/pretrained_model.pt` 
