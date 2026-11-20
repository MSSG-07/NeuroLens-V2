# NeuroLens V2
### 3D-to-2D Generalization in Brain MRI

NeuroLens V2 is a research project investigating whether knowledge learned by a 3D teacher model can be transferred to a 2D student model for brain MRI analysis.

The project explores whether a student model trained with teacher-guided knowledge distillation can perform classification, tumor segmentation, and normalized spatial depth estimation using individual 2D MRI slices.

## Research Objectives

- Investigate 3D-to-2D knowledge transfer in brain MRI.
- Evaluate isolated 2D slice inference without neighbouring slices or 3D volume information.
- Perform multi-task learning for classification, segmentation, and depth estimation.
- Compare student performance against the 3D teacher model.

## Model Overview

**3D Teacher:** Processes volumetric MRI data containing four modalities.

**2D Student:** Processes individual four-channel MRI slices independently.

The teacher guides the student during training through knowledge distillation.

## Dataset

The research uses the BraTS 2023 Adult Glioma dataset.

Four MRI modalities are used:

- T1
- T1 contrast-enhanced (T1c)
- T2
- FLAIR

The dataset is split at the patient level into training, validation, and testing subsets to prevent patient overlap.

Dataset files are not included in this repository. Refer to the dataset setup documentation for the expected structure.

## Repository Contents

- `notebooks/` — Research and experimental notebooks.
- `docs/` — Methodology, dataset setup, and experiment documentation.
- `data/` — Expected dataset and manifest directory structure.
- `outputs/` — Location for generated experiment outputs.

## Evaluation Tasks

1. Brain MRI classification.
2. Tumor segmentation.
3. Normalized spatial depth estimation.

## Reproducibility

The notebook is designed for Google Colab. Users must configure their own dataset paths and install the required dependencies before running experiments.

## Limitations

- Depth estimation represents normalized relative position, not physical distance in millimetres.
- The student expects four MRI modalities.
- NeuroLens V2 is a research prototype and is not a clinically validated diagnostic system.

## Research Status

This repository accompanies the NeuroLens V2 experimental study. Results and conclusions should be interpreted within the scope of the evaluated dataset and experimental setup.
