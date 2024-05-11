# Fake News Detection

Fake News Detection in Python

In this project, we have used various natural language processing techniques and machine learning algorithms to classify fake news articles using sci-kit libraries from python. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them:

1. Python 3.12 
   - This setup requires that your machine has python 3.12 installed on it.
2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands. To install anaconda check this url https://www.anaconda.com/download/
3. You will also need to download and install below 3 packages after you install either python or anaconda from the steps above
   - Sklearn (scikit-learn)
   - numpy
   - scipy
   
  - if you have chosen to install python 3.12 then run below commands in command prompt/terminal to install these packages
   ```
   pip install -U scikit-learn
   pip install numpy
   pip install scipy
   ```
   - if you have chosen to install anaconda then run below commands in anaconda prompt to install these packages
   ```
   conda install -c scikit-learn
   conda install -c anaconda numpy
   conda install -c anaconda scipy
   ```
#### Dataset used
The data source used for this project is Fake and Real News dataset which contains 2 files with .csv format for test, train. Below is some description about the data files used for this project.
FAKE AND REAL NEWS DATASET
This dataset contains two file named as true.csv and fake.csv .
The dataset contain 4 columns for train and test.
Dataset columns:

1.Title: title of news article
2.Text: body text of news article
3.Subject: subject of news article
4.Date: publish date of news article

### File descriptions

#### main.py
This file contains all the pre processing functions needed to process all input documents and texts. First we read the train and test data files then performed some pre processing like tokenizing, stemming etc. 
In this file we have also performed feature extraction and selection methods from sci-kit learn python libraries. For feature selection, we have used methods like simple bag-of-words and n-grams and then term frequency like countvectorizer.
Here we have also build the classifiers for predicting the fake news detection. The extracted features are fed into logistic regression classifiers. 
#### app.py
Our best performing classifier was ```Logistic Regression``` which was then saved on disk with name ```model.joblib```. Once you close this repository, this model will be copied to user's machine and will be used by app.py file to classify the fake news. It takes an news article as input from user then model is used for final classification output that is shown to user along with whether it is real or fake.
	
