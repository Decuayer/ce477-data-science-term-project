# AFAD Seismic Activity Analysis and Data Science Pipeline

**Course:** CE 477 – Data Science  
**Institution:** Izmir University of Economics  
**Department:** Department of Computer Engineering  

---

## 1. Project Overview

This repository contains the end-to-end data science project developed for the CE 477 Data Science course at Izmir University of Economics. The project investigates seismic activity records across Turkey and adjacent regions provided by the Republic of Turkey Ministry of Interior Disaster and Emergency Management Authority (AFAD). 

The primary objective is to build a reproducible, robust data pipeline that ingests raw seismic records, performs comprehensive exploratory data analysis (EDA), executes preprocessing and feature engineering workflows, and prepares balanced representations for downstream predictive modeling.

---

## 2. Repository Structure

```text
.
├── data/
│   ├── afad_earthquake_data_raw.csv           # Raw seismic dataset retrieved from AFAD
│   └── afad_earthquake_data_preprocessed.csv  # Cleaned, encoded, and preprocessed dataset
├── figures/                                   # Exported visualizations and analytical plots
│   ├── fig1_magnitude_type_distribution.png
│   ├── fig2_numeric_boxplots.png
│   ├── fig3_depth_mag_boxplot.png
│   ├── fig4_histograms.png
│   ├── fig5_spatial_scatter.png
│   ├── fig6_depth_vs_magnitude.png
│   ├── fig7_temporal_lineplot.png
│   ├── fig8_correlation_matrix.png
│   ├── fig9_outlier_filtered_boxplots.png
│   ├── fig10_discretized_magnitude_classes.png
│   ├── fig11_pca_2d_projection.png
│   ├── fig12_pca_3d_projection.png
│   └── fig13_data_augmentation_smote.png
├── notebooks/
│   ├── 00_afad_earthquake_data_collection.ipynb  # Data retrieval pipeline (API and cloud fallback)
│   └── 01_dataset_analysis.ipynb                 # EDA, statistical analysis, and preprocessing
├── .gitignore                                 # System and environment exclusions
└── README.md                                  # Project documentation
```

---

## 3. Methodological Workflow

### 3.1. Data Ingestion and Pipeline Architecture
The collection stage (`notebooks/00_afad_earthquake_data_collection.ipynb`) connects to the AFAD Earthquake Department API with automated date-chunking to bypass query payload constraints. It incorporates fallback mechanisms to ensure complete retrieval of historical seismic observations.

### 3.2. Exploratory Data Analysis (EDA)
The statistical and descriptive inspection (`notebooks/01_dataset_analysis.ipynb`) covers:
- **Dimensionality and Metadata Inspection:** Structural checks and type verification across spatial, temporal, and magnitude attributes.
- **Missing Value Auditing:** Quantitative analysis of completeness across features.
- **Categorical and Ordinal Distributions:** Evaluation of magnitude measurement types (e.g., $M_w$, $M_l$) and geographical event classifications.
- **Dispersion and Dispersion Analytics:** Interquartile range (IQR) boxplots and continuous frequency histograms evaluating earthquake depth and magnitude.
- **Bivariate and Spatial Relationships:** Depth-magnitude interaction scatter plots and geographical coordinate mappings.
- **Temporal Trends:** Longitudinal event frequencies and seasonal variance tracking.

### 3.3. Data Preprocessing and Feature Engineering
- **Multicollinearity Assessment:** Feature correlation matrix estimation via heatmaps.
- **Outlier Mitigation:** Domain-informed boundary filtering to remove corrupted and unphysical instrument readings.
- **Attribute Discretization:** Transformation of continuous magnitude values into standardized earthquake severity intervals.
- **Scaling and Encoding:** Implementation of Min-Max normalization, Z-score standardization, and One-Hot Encoding for categorical variables.
- **Dimensionality Reduction:** 2D and 3D Principal Component Analysis (PCA) to evaluate variance explained and feature separability.
- **Class Balancing:** Application of Synthetic Minority Over-sampling Technique (SMOTE) to address severe frequency imbalances across high-magnitude classes.

---

## 4. Environment and Setup

### Prerequisites
- Python 3.10 or higher
- Jupyter Notebook / JupyterLab environment

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Decuayer/ce477-data-science-term-project.git
   cd ce477-data-science-term-project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn requests
   ```

---

## 5. Execution

The analysis and preprocessing stages can be reproduced sequentially:

1. **Data Ingestion:**
   Open and execute `notebooks/00_afad_earthquake_data_collection.ipynb` to regenerate or verify the raw dataset.
2. **Analysis and Preprocessing:**
   Execute `notebooks/01_dataset_analysis.ipynb` to reproduce the statistical figures in `figures/` and export `data/afad_earthquake_data_preprocessed.csv`.
