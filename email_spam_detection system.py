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


"""
 function that generates a dictionary, recording the frequency with which each word in the dataset appears as spam (1) or ham (0)."""
def get_word_frequency(X, Y) :
    """
    Calculate the frequency of each word in a set of emails categorized as spam (1) or not spam (0).

    Parameters:
    - X (numpy.array): Array of emails, where each email is represented as a list of words.
    - Y (numpy.array): Array of labels corresponding to each email in X. 1 indicates spam, 0 indicates ham.

    Returns:
    - word_dict (dict): A dictionary where keys are unique words found in the emails, and values
      are dictionaries containing the frequency of each word for spam (1) and not spam (0) emails.
    """
    # Creates an empty dictionary
    word_dict = {}

    ### START CODE HERE ###

    num_emails = len(X)

    # Iterates over every processed email and its label
    for i in range(num_emails) :
        # Get the i-th email
        email = X[i]
        # Get the i-th label. This indicates whether the email is spam or not. 1 = None
        # The variable name cls is an abbreviation for class, a reserved word in Python.
        cls = Y[i]
        # To avoid counting the same word twice in an email, remove duplicates by casting the email as a set
        email = set(email)
        # Iterates over every distinct word in the email
        for word in email :
            # If the word is not already in the dictionary, manually add it. Remember that you will start every word count as 1 both in spam and ham
            if word not in word_dict.keys() :
                word_dict[word] = {"spam" : 1, "ham" : 1}
            # Add one occurrence for that specific word in the key ham if cls == 0 and spam if cls == 1.
            if cls == 0 :
                word_dict[word]["ham"] += 1
            if cls == 1 :
                word_dict[word]["spam"] += 1

    ### END CODE HERE ###
    return word_dict


# test the above  function
test_output = get_word_frequency([['like','going','river'], ['love', 'deep', 'river'], ['hate','river']], [1,0,0])
print(test_output)

# word frequency
word_frequency = get_word_frequency(X_train,Y_train)
# To count the spam and ham emails, you may just sum the respective 1 and 0 values in the training dataset, since the convention is spam = 1 and ham = 0.
class_frequency = {'ham': sum(1 for i in Y_train if i == 0 ), 'spam': sum(1 for j in Y_train if j == 1)}
print(class_frequency)

"""
will implement the function to compute  𝑃(word∣spam)
  and  𝑃(word∣ham)
 . Since the computations are the same for both types of emails, you will create a function to compute  𝑃(word∣class)
  where class can be either spam ( 1
 ) or (ham)  0
 .

Remember that

𝑃(word𝑖∣class)=\# emails in the class (either spam or ham) containing word𝑖\# emails in the given class (spam or ham)
"""


def prob_word_given_class(word, cls, word_frequency, class_frequency) :
    """
    Calculate the conditional probability of a given word occurring in a specific class.

    Parameters:
    - word (str): The target word for which the probability is calculated.
    - cls (str): The class for which the probability is calculated, it may be 'spam' or 'ham'
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.

    Returns:
    - float: The conditional probability of the given word occurring in the specified class.
    """
    ### START CODE HERE ###

    # Get the amount of times the word appears with the given class (class is stores in spam variable)
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class / class_frequency[cls]

    ### END CODE HERE ###
    return p_word_given_class

# sample of th current value of he

print(f"P(lottery | spam) = {prob_word_given_class('lottery', cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(lottery | ham) = {prob_word_given_class('lottery', cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(schedule | spam) = {prob_word_given_class('schedule', cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(schedule | ham) = {prob_word_given_class('schedule', cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)}")

def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Calculate the probability of an email being of a certain class (e.g., spam or ham) based on treated email content.

    Parameters:
    - treated_email (list): A list of treated words in the email.
    - cls (str): The class label for the email. It can be either 'spam' or 'ham'
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.

    Returns:
    - float: The probability of the given email belonging to the specified class.
    """

    # prob starts at 1 because it will be updated by multiplying it with the current P(word | class) in every iteration
    prob = 1

    ### START CODE HERE ###

    for word in treated_email:
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequency.keys():
            # Update the prob by multiplying it with P(word | class). Don't forget to add the word_frequency and class_frequency parameters!
            prob *= prob_word_given_class(word, cls = cls, word_frequency = word_frequency, class_frequency = class_frequency)

    return prob

example_email = "Click here to win a lottery ticket and claim your prize!"
treated_email = preprocess_text(example_email)

# this computes probability of an email given spam
prob_spam = prob_email_given_class(treated_email, cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)

# this computes probability of an email given spam
prob_ham = prob_email_given_class(treated_email, cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)
print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nP(email | spam) = {prob_spam}\nP(email | ham) = {prob_ham}")

