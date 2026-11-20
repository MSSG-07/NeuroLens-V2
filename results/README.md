# NeuroLens V2 – Experimental Results

This directory contains the experimental results generated during the evaluation of **NeuroLens V2: 3D-to-2D Generalization for Brain Tumor Classification, Segmentation, and Spatial Depth Estimation**.

The CSV files preserve the quantitative evaluation outputs from the research pipeline and support comparison between the 3D teacher model, the isolated 2D student model, and external dataset evaluation.

## Results Overview

| File | Description |
|---|---|
| `final_comparison.csv` | Consolidated comparison of the experimental results. Provides an overview of the evaluated approaches and their outcomes. |
| `3d_results.csv` | Evaluation results from the 3D teacher model operating on volumetric MRI data. |
| `isolated_2d_results.csv` | Evaluation results for the 2D student model using individual MRI slices independently. |
| `brisc_results.csv` | Results from the external evaluation using the BRISC 2025 dataset. |
| `depth_results.csv` | Quantitative results for the spatial depth-estimation task. |

## Research Evaluation

The results correspond to the following research objectives:

### 1. 3D Teacher Evaluation
Evaluates the teacher model trained using volumetric MRI data. The results are recorded in `3d_results.csv`.

### 2. Isolated 2D Student Evaluation
Examines whether the 2D student model can perform the intended prediction tasks using individual MRI slices rather than complete 3D volumes. The results are recorded in `isolated_2d_results.csv`.

### 3. External Dataset Evaluation
Evaluates model performance using BRISC 2025 as an independent external 2D dataset. The results are recorded in `brisc_results.csv`.

### 4. Spatial Depth Estimation
Stores the quantitative outputs associated with estimating normalized axial spatial depth. The results are recorded in `depth_results.csv`.

### 5. Final Experimental Comparison
`final_comparison.csv` provides a consolidated view of the experimental findings across the evaluated approaches.

## Data Format

All files are provided in CSV (Comma-Separated Values) format. They can be opened using Python, Pandas, Microsoft Excel, or other spreadsheet applications.

Example:

```python
import pandas as pd

results = pd.read_csv("results/final_comparison.csv")
print(results.head())
```

## Reproducibility

These files contain experimental outputs and are provided to support inspection and comparison of the reported results.

For the corresponding model architecture, training pipeline, evaluation procedures, and experiment configuration, refer to the source code and notebooks in the main repository.

The MRI datasets are not included in this repository. Refer to the main `README.md` for dataset access information and the required data organisation.

## Notes

- Results represent the outputs of the experiments conducted for this research project.
- CSV files should be interpreted in the context of their corresponding evaluation procedures.
- The consolidated comparison should not be treated as a substitute for the individual evaluation files.
- No raw MRI scans are distributed through this results directory.
