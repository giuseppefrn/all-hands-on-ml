# All Hands on ML

A comprehensive repository of machine learning examples and tutorials designed to walk through various machine learning concepts, techniques, and applications.

## Project Description

"All Hands on ML" is a collection of Jupyter notebooks and Python scripts that demonstrate machine learning workflows from data preprocessing to model deployment. This repository is intended for:

- Beginners looking to learn machine learning concepts
- Intermediate practitioners seeking practical examples
- Anyone interested in hands-on machine learning tutorials

Each example is designed to be self-contained and educational, with detailed explanations and comments throughout the code.

## Project Structure

```
all-hands-on-ml/
├── introduction/
│   └── titanic_data_preprocessing.ipynb - Data preprocessing example using the Titanic dataset
├── README.md - This file
├── pyproject.toml - Project configuration and dependencies
└── poetry.lock - Locked dependencies for reproducibility
```

More examples will be added in the future, covering topics such as:
- Feature engineering
- Model selection and evaluation
- Deep learning
- Natural language processing
- Computer vision
- And more!

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

Navigate to the notebook you want to run, for example:
```
introduction/titanic_data_preprocessing.ipynb
```

### Example: Titanic Data Preprocessing

The introductory notebook demonstrates:
- Loading and exploring the Titanic dataset
- Visualizing key features
- Handling missing values
- Feature engineering
- Preparing data for machine learning models

## Dependencies

The project relies on the following main libraries:
- pandas (>=2.2.3,<3.0.0)
- matplotlib (>=3.10.3,<4.0.0)
- seaborn (>=0.13.2,<0.14.0)
- notebook (>=7.4.2,<8.0.0)

All dependencies are managed through Poetry and specified in the `pyproject.toml` file.

## Contributing

Contributions are welcome! If you'd like to add examples, fix bugs, or improve documentation, please feel free to submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).