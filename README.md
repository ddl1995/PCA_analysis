# # PCA of Water Quality Parameters (Generic)

This script performs Principal Component Analysis (PCA) on site-based parameter data
and visualizes the PCA scores and loadings. PCA is computed only from parameter columns
(`parameter1`, `parameter2`, ...). `WQI` (or any numeric index) is used only for coloring
the PCA score plot.

## Input CSV format
Required columns:
- Site ID
- WQI
- parameter1, parameter2, ... (numeric)

Example header:
Site ID,WQI,parameter1,parameter2,parameter3,parameter4,parameter5,parameter6

## Install
```bash
pip install -r requirements.txt
