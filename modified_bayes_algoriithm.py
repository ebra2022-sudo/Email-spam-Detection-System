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


def log_naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood=False) :
    """
    Naive Bayes classifier for spam detection, comparing the log probabilities instead of the actual probabilities.

    This function calculates the log probability of an email being spam (1) or ham (0)
    based on the Naive Bayes algorithm. It uses the conditional probabilities of the
    treated_email given spam and ham, as well as the prior probabilities of spam and ham
    classes. The final decision is made by comparing the calculated probabilities.

    Parameters:
    - treated_email (list): A preprocessed representation of the input email.
    - return_likelihood (bool): If true, it returns the log_likelihood of both spam and ham.

    Returns:
    - int: 1 if the email is classified as spam, 0 if classified as ham.
    """

    # Compute P(email | spam) with the new log function
    log_prob_email_given_spam = log_prob_email_given_class(treated_email, cls='spam', word_frequency=word_frequency,
                                                           class_frequency=class_frequency)

    # Compute P(email | ham) with the function you defined just above
    log_prob_email_given_ham = log_prob_email_given_class(treated_email, cls='ham', word_frequency=word_frequency,
                                                          class_frequency=class_frequency)

    # Compute P(spam) using the class_frequency dictionary and using the formula #spam emails / #total emails
    p_spam = class_frequency['spam'] / (class_frequency['ham'] + class_frequency['spam'])

    # Compute P(ham) using the class_frequency dictionary and using the formula #ham emails / #total emails
    p_ham = class_frequency['ham'] / (class_frequency['ham'] + class_frequency['spam'])

    # Compute the quantity log(P(spam)) + log(P(email | spam)), let's call it log_spam_likelihood
    log_spam_likelihood = np.log(p_spam) + log_prob_email_given_spam

    # Compute the quantity P(ham) * P(email | ham), let's call it ham_likelihood
    log_ham_likelihood = np.log(p_ham) + log_prob_email_given_ham

    # In case of passing return_likelihood = True, then return the desired tuple
    if return_likelihood == True :
        return (log_spam_likelihood, log_ham_likelihood)

    # Compares both values and choose the class corresponding to the higher value.
    # As the logarithm is an increasing function, the class with the higher value retains this property.
    if log_spam_likelihood >= log_ham_likelihood :
        return 1
    else :
        return 0