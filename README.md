# DSND Pipeline Project

This repository contains all the needed code, data and results for the DSND Pipeline Project. The aim of this project is to create a customer recommendation based on their reviews using pipelines, Natural Language Processing techniques and machine learning models.

## Requirements

* Python 3.x
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/)
* [spaCy](https://spacy.io/)
* [Jupyter Notebook](https://jupyter.org/)

## Installation
**Install the required packages:**
```
pip install pandas scikit-learn spacy notebook
pyhon -m spacy download en_core_web_sm
```

## Usage
1. Open a Jupyter Notebook file
2. Run the notebook cells in order to:
    - Load and explore the dataset.
    - Preprocess the data
    - Build and train the model pipeline
    - Fine-tune the model using RandomizedSearchCV
    - Evaluate the final model using test data

## Data Details
The provided dataset includes:
- Numerical Features: `Age`, `Positive Feedback Count`
- Categorical Features: `Division Name`, `Department Name`, `Class Name`
- Text Features: `Title`, `Review Text` [these are combined together to make a new column called `Full Review`]
    - These coluns [`Title` and  `Review Text`] are dropped from the table


# Features
- Data Exploration
    Inital data loading and exploration to understand the dataset
- Preprocessing (or Feature Engineering)
    - Numerical data using `SimpleImputer` and `MinMaxScaler`
    - Categorical data using `OrdinalEncoder`, `SimpleImputer` and `OneHotEncoder`
    - Text data using spaCy to tokenize and TF-IDF vectorization i.e., `TfidfVectorizer`
- Model Training and Fine-Tuning:
    - Used DecisionTreeClassifier model to build the pipeline.
    - Tuning hyperparameter using `GridSearchCV`
        - The following parameters are used to fine-tune the model
            - decisiontreeclassifier__max_depth
            - decisiontreeclassifier__max_features
            - decisiontreeclassifier__splitter
            - decisiontreeclassifier__random_state
    - Displays the best hyperparameter used in fine-tuning the model
- Evaluation
    - The final model is evaluated with an accuarcy metrics

## License

This project is part of Udacity coursework and follows their guidelines
