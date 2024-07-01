# Depop Test

# Requirements

1. Install Python dependencies with [Miniconda](https://docs.anaconda.com/miniconda/):
```
conda env create --name depop_test_3 --file=env.yml
```
2. Activate the environment
```
conda activate depop_test_3
```
3. Download `en_core_web_sm`
```
python -m spacy download en_core_web_sm
```
4. Launch `main.py` file to start data preprocessing, training process and metrics evaluation
```
python main.py
```

# Project Structure

This section outlines the structure of the project and the purpose of each directory and file.



```
depop-test-3/
├── config.py
├── feature_extractor.py
├── main.py
├── nlp_singleton.py
├── README.md
├── LICENSE
├── data/
│   ├── test_set.tsv
│   └── train_set.tsv
├── utils.py
└── env.yml
```

# Files and Directories
- `config.py` : Contains configuration classes used throughout the project, including dataset paths, feature extraction parameters, and classifier settings.
- `feature_extractor.py`: Implements functions for extracting features from text data.
-`main.py`: The main script to run the entire process of data loading, preprocessing, feature extraction, model training, and evaluation.
- `README.md`: Provides an overview of the project, setup instructions, and usage information.
- `utils.py`: Contains utility functions for preprocessing and evaluating text data.
- `env.yml`: YAML file for setting up the conda environment.
- `nlp_singleton.py`: file containing NLP Singleton used for NLP text preprocessing
- `LICENCE`: MIT Licence file 
- `data/`: Folder containing training and test data
- - `train_set.tsv`: Tab separated file containing train dataset
- - `test_set.tsv`: Tab separated file containing test dataset

# Configuration

`config.py` file contains three main configuration classes:
- `ProjectStructure`: Defines the directory structure for the project.
- `DatasetConfig`: Specifies dataset paths, column names, and other dataset-related configurations.
- `FeatureExtractor`: Contains parameters for feature extraction, such as the number of features to extract and the language of the text.
- `Classifier`: Holds the parameters for the classifier, including learning rate, maximum depth, number of estimators, and more.

# Contributing
If you want to contribute to this project, please fork the repository and submit a pull request. Ensure that your code follows the project's coding standards and includes appropriate tests.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.