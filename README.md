# Machine-Learning-Based-Prediction-of-Vinho-Verde-Red-Wine-Quality
Project to predict Vinho Verde red wine quality using a machine-learning pipeline: preprocessing, feature engineering and PCA on physicochemical variables; compared SVM, k-NN and decision trees with cross-validation and model selection, and documented preprocessing steps, hyperparameter tuning, evaluation metrics and reproducible code.

## Project Summary
This repository contains all resources for a practical project focused on predicting the quality of *Vinho Verde* red wine using machine learning techniques. The study employs physicochemical variables as predictors, performing preprocessing and data cleaning (in R), dimensionality reduction through PCA, and classification experiments implemented in Julia. The full practicum report is included for further methodological details and results.

## Repository Contents
```
AA1_Vino/
├── data/
│   ├── winequality-red.data
│   ├── cv_indices.dat
│   └── preprocess_vino.R
├── src/
│   ├── functions.jl
│   ├── P2.jl
│   └── soluciones.jl
├── report/
│   └── AA1_VinhoVerde.pdf
README.md
```

## File Descriptions
- `AA1_Vino/src/functions.jl` — collection of helper functions used throughout the experiments.  
- `AA1_Vino/src/P2.jl` — main Julia script and experiment runner; executes model training, validation, and evaluation.  
- `AA1_Vino/src/soluciones.jl` — complementary Julia script including provided solutions and validation functions, used as reference for correctness.  
- `AA1_Vino/data/winequality-red.data` — dataset containing physicochemical measurements and quality scores for red *Vinho Verde* wines.  
- `AA1_Vino/data/preprocess_vino.R` — R script responsible for preprocessing, cleaning, and preparing the dataset for model training.  
- `AA1_Vino/data/cv_indices.dat` — precomputed indices used for 10-fold cross-validation to ensure consistent and reproducible results.  
- `AA1_Vino/report/AA1_VinhoVerde.pdf` — detailed practicum report explaining the data analysis, PCA, algorithms, results, and conclusions.

## Requirements
To reproduce this project, the following environments and dependencies are recommended:

### Software
- **Julia** 1.x — for all model implementation and analysis.  
- **R** ≥ 3.6 — to execute data preprocessing (`preprocess_vino.R`).  

### Julia Environment
Ensure the necessary packages are installed as listed at the beginning of the Julia scripts (look for `using` statements).  
If working within a Julia environment, create and activate a `Project.toml` and `Manifest.toml` for reproducibility:

```bash
julia --project=.
Pkg.instantiate()
```

### R Environment
The preprocessing script can be executed directly with Rscript. Recommended R packages (if required) are specified at the beginning of the `preprocess_vino.R` script.

## Quick Start — Reproduce the Analysis

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Preprocess the Dataset (R)**
   ```bash
   # from repository root
   Rscript "AA1_Vino/data/preprocess_vino.R"
   ```
   This step cleans and prepares the dataset for further analysis, generating any intermediate files required by the Julia scripts.

3. **Run the Julia Experiments**
   ```bash
   # execute main script
   julia "AA1_Vino/src/P2.jl"
   ```
   If you are using a Julia environment, specify it explicitly:
   ```bash
   julia --project="AA1_Vino/src" "AA1_Vino/src/P2.jl"
   ```
   The main script performs PCA, trains multiple models, applies 10-fold cross-validation, and outputs performance metrics.

4. **View Results**
   The generated results, performance metrics, and any output figures can be inspected directly or reviewed in the accompanying report:
   `AA1_Vino/report/AA1_VinhoVerde.pdf`.

## Methodology Overview
The project follows a complete machine learning workflow, including:

1. **Data Preprocessing:**  
   Cleaning, normalization, and outlier handling using R scripts.  

2. **Dimensionality Reduction:**  
   Application of Principal Component Analysis (PCA) to identify the most informative features.  

3. **Model Training and Evaluation:**  
   Implementation and comparison of multiple classification algorithms using Julia, including Artificial Neural Networks (ANN), Support Vector Machines (SVC), Decision Trees, and k-Nearest Neighbors (k-NN).  

4. **Cross-Validation:**  
   10-fold cross-validation is used to validate model stability and ensure reproducible results.  

5. **Performance Metrics:**  
   Models are compared based on accuracy, F1-score, and confusion matrix analysis to evaluate predictive performance.

## Reproducibility Notes
- The project includes a predefined split file (`cv_indices.dat`) to ensure the same data partitions are used across experiments.  
- PCA components and selected features are documented in the practicum report.  
- Model parameters, hyperparameter tuning processes, and evaluation procedures are explained in detail within the report for transparency and repeatability.

## Results Summary
According to the practicum report:
- The Support Vector Classifier (SVC) achieved the highest predictive accuracy (around 77% in 10-fold cross-validation).  
- Artificial Neural Networks (ANN) reached an average accuracy of approximately 72%.  
- Decision Trees and k-NN achieved comparable but slightly lower accuracies.  
- The models demonstrated consistent performance and reliability, confirming the importance of PCA preprocessing and proper normalization.

## Author
Developed by **Jorge Crespo Rivas** and **Álvaro Sieira Rama**.  
Universidade da Coruña — Bachelor’s Degree in Data Science and Engineering (2025).
