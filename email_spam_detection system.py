# pandas library for dataset manipulation
"""
Purpose: Data Handling & Analysis
Usage in Spam Detection:

Reads and processes email datasets (usually in CSV format).

Organizes email text and labels (spam vs. non-spam) into structured tables.

Enables efficient data manipulation (e.g., filtering, grouping, cleaning).
"""
import os.path

import nltk
import pandas as pd

# numpy library for numerical computation
"""
Purpose: Numerical Computation
Usage in Spam Detection:

Efficiently handles large arrays and matrices (useful for storing feature vectors).

Often used in machine learning models for computations.

Helps in handling vectorized operations when transforming text into numerical representations (e.g., TF-IDF, word embeddings).
"""
import numpy as np

# nlkt(Natural language toolkit)
"""
Purpose: Removes commonly used words (stopwords) like "the," "and," "is," which do not contribute to spam detection.

Usage in Spam Detection:

Improves the accuracy of models by reducing noise in text data.

Helps focus on important words that indicate spam (e.g., "win," "free," "click").
"""
from nltk.corpus import  stopwords
"""
This are used to download the nlkt's pretrained model and data set to perform the operation 
since it is impossible to download with pip installation"""
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# nlkt(Natural language toolkit)
"""
Purpose: Splits email text into individual words (tokens).

Usage in Spam Detection:

Helps in feature extraction by breaking down an email into meaningful words.

Essential for applying further processing like stemming, lemmatization, or TF-IDF vectorization.
"""
from nltk import word_tokenize

# string library
"""
Purpose: Handles text cleaning by removing punctuation.
Usage in Spam Detection:

Spam emails may contain extra symbols (e.g., "!!!FREE!!!") to evade filters.

Removing punctuation helps in standardizing the text for analysis.
"""
import  string


#load dataset using panda stored in the ./data directory
dataframe_emails = pd.read_csv("data/emails.csv")

# overview of dataset
print(dataframe_emails.columns)
print(f"Number of emails: {len(dataframe_emails)}")
print(f"Proportion of spam emails: {dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(f"Proportion of ham emails: {1-dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(dataframe_emails.head())


def preprocess_dataset(df):
    # shuffle the dataset to avoid biasing due to ordering of the dataset
    df = df.sample(frac = 1.0, random_state =42, ignore_index = True)
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    Y = df.spam.to_numpy()

    return X,Y
X, Y = preprocess_dataset(dataframe_emails) # X-> The text column array, Y-> the spam column array
print(X)
print(Y)

# preprocess the text by removing stopwords and punctuation using nltk library

def preprocess_text(text):
    # to make a set of english words and punctuations to fiter out from text of the email
    stop = set(stopwords.words('english') + list(string.punctuation))
    stop = set(stopwords.words('english') + list(string.punctuation))

    if isinstance(text, str):
        text = np.array([text])

    # initialize the preprocessed text
    text_preprocessed = []

    """
    This code traverse through the array of email text, filter out stop words and store in text_preprocessed array of numpy array of string
    """
    for _, email in enumerate(text) :
        email = np.array([word.lower() for word in word_tokenize(email) if word.lower() not in stop]).astype(X.dtype)
        text_preprocessed.append(email)

    if len(text) == 1 :
        return text_preprocessed[0]
    return text_preprocessed

X_treated = preprocess_text(X)
email_index = 989
print(f"Email before preprocessing: {X[email_index]}")
print(f"Email after preprocessing: {X_treated[email_index]}")

"""
Splitting into train/test
Now let's split our dataset into train and test sets. You will work with a proportion of 80/20,
i.e., 80% of the data will be used for training and 20% for testing.
"""

train_size = int(0.8 * len(X_treated)) # 80 % for training and the remaining for testing

X_train, X_testt = X_treated[:train_size], X_treated[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

"""
Remember that there about 24% of the emails are spam.
 It is important to check if this proportion remains roughly
 the same in the train and test datasets, otherwise you may build a biased algorithm
"""
print(f"Proportion of spam in train dataset: {sum(i for i in Y_train if i== 1)/len(Y_train):.4f}") # 0.2431(of train dataset) ~ 0.2388(of total dataset)
print(f"Proportion of spam in test dataset: {sum(i for i in Y_test if i== 1)/len(Y_test):.4f}") # 0.0.2216(of test dataset) ~ 0.2388(of total dataset)



