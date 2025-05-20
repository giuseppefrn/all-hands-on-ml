# All Hands on ML

A comprehensive repository of machine learning examples, tutorials, and deployment solutions designed to provide hands-on experience with various ML concepts, techniques, and applications.

## Project Description

"All Hands on ML" is a collection of Jupyter notebooks and Python scripts that demonstrate complete machine learning workflows from data preprocessing to model deployment. This repository is intended for:

- Beginners looking to learn machine learning concepts
- Intermediate practitioners seeking practical examples
- Advanced users interested in deployment solutions
- Anyone wanting to explore hyperparameter optimization and experiment tracking

Each example is designed to be educational, with detailed explanations and comments throughout the code.

## Project Structure

```
all-hands-on-ml/
├── data/
│   ├── raw/ - Original, unmodified datasets
│   └── processed/ - Cleaned and preprocessed datasets
├── src/ - Source code and notebooks
│   ├── 1_titanic_data_preprocessing.ipynb - Data preprocessing example
│   ├── 2_logistic_regression_titanic.ipynb - Basic classification model
│   ├── 3_decision_tree_titanic.ipynb - Decision tree implementation
│   ├── 4_random_forest_titanic.ipynb - Ensemble methods
│   ├── 5_normalize_and_split_titanic.ipynb - Data normalization techniques
│   ├── 6_logistic_regression_titanic.ipynb - Advanced logistic regression
│   ├── 7_svm_logreg_dtree_comparison.py - Model comparison
│   ├── 8_xgboost_and_others_comparison.py - Advanced model comparison
│   ├── 9_kmeans_dbscan_comparison_notebook.py - Clustering algorithms
│   ├── 10_regression_models_comparison.py - Regression techniques
│   ├── 11_anomaly_detection_ocsvm_iforest.py - Anomaly detection
│   └── 12_gridsearch_optuna_mlflow_demo.py - Hyperparameter optimization
├── deployment/ - Model deployment examples
│   ├── fastapi/ - FastAPI deployment solution
│   └── mlflow/ - MLflow deployment solution
├── artifacts/ - Saved models and other artifacts
├── README.md - This file
├── pyproject.toml - Project configuration and dependencies
└── poetry.lock - Locked dependencies for reproducibility
```

## Key Features

- **Data Preprocessing**: Techniques for cleaning, transforming, and preparing data
- **Model Training**: Implementation of various ML algorithms (classification, regression, clustering)
- **Model Comparison**: Comparative analysis of different algorithms on the same datasets
- **Hyperparameter Optimization**: Using GridSearchCV and Optuna for tuning models
- **Experiment Tracking**: Integration with MLflow for tracking experiments
- **Model Deployment**: Deployment solutions using FastAPI and MLflow with Docker

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

- Python 3.13 or higher
- Poetry

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/all-hands-on-ml.git
   cd all-hands-on-ml
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Usage

### Running Jupyter Notebooks

After activating the Poetry environment:

```bash
jupyter notebook
```

Navigate to the notebook you want to run in the `src` directory.

### Running Python Scripts

After activating the Poetry environment:

```bash
python src/script_name.py
```

For example:
```bash
python src/12_gridsearch_optuna_mlflow_demo.py
```

### Model Deployment

#### FastAPI Deployment

1. Train and save your model
2. Use the provided FastAPI script in `deployment/fastapi/`
3. Build and run the Docker container:
   ```bash
   cd deployment/fastapi
   docker build -t fastapi-model-server .
   docker run --rm -p 8000:8000 fastapi-model-server
   ```
4. Access the API at http://0.0.0.0:8000 or the Swagger UI at http://0.0.0.0:8000/docs

#### MLflow Deployment

1. Train and save your model with MLflow
2. Use the provided Dockerfile in `deployment/mlflow/`
3. Build and run the Docker container:
   ```bash
   cd deployment/mlflow
   docker build -t mlflow-model-server .
   docker run --rm -p 1234:1234 mlflow-model-server
   ```
4. Send prediction requests to http://localhost:1234/invocations

## Dependencies

The project relies on the following main libraries:
- pandas (>=2.2.3,<3.0.0)
- matplotlib (>=3.10.3,<4.0.0)
- seaborn (>=0.13.2,<0.14.0)
- scikit-learn (>=1.6.1,<2.0.0)
- xgboost (>=3.0.1,<4.0.0)
- mlflow (>=2.22.0,<3.0.0)
- fastapi (>=0.115.12,<0.116.0)
- optuna (>=4.3.0,<5.0.0)

All dependencies are managed through Poetry and specified in the `pyproject.toml` file.

## Contributing

Contributions are welcome! If you'd like to add examples, fix bugs, or improve documentation, please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
