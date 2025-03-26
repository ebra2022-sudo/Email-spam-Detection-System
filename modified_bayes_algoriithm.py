import numpy as np

from email_spam_detection_system import prob_word_given_class


def log_prob_email_given_class(treated_email, cls, word_frequency, class_frequency) :
    """
    Calculate the log probability of an email being of a certain class (e.g., spam or ham) based on treated email content.

    Parameters:
    - treated_email (list): A list of treated words in the email.
    - cls (str): The class label ('spam' or 'ham')


    Returns:
    - float: The log probability of the given email belonging to the specified class.
    """

    # prob starts at 0 because it will be updated by summing it with the current log(P(word | class)) in every iteration
    prob = 0

    for word in treated_email :
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequency.keys() :
            # Update the prob by summing it with log(P(word | class))
            prob += np.log(prob_word_given_class(word, cls, word_frequency, class_frequency))

    return prob