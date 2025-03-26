import numpy as np

from email_spam_detection_system import prob_word_given_class, treated_email, word_frequency, class_frequency, Y, \
    Y_test, X_test
from limitation_of_naive_bayes_algorithm import example_index
from model_performance import get_true_positives, get_true_negatives


def log_prob_email_given_class(treated_emails, cls, word_frequencies, class_frequencies) :
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

    for word in treated_emails :
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequencies.keys() :
            # Update the prob by summing it with log(P(word | class))
            prob += np.log(prob_word_given_class(word, cls, word_frequencies, class_frequencies))

    return prob


def log_naive_bayes(treated_emails, word_frequencies, class_frequencies, return_likelihood=False) :
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
    log_prob_email_given_spam = log_prob_email_given_class(treated_emails, cls='spam', word_frequencies=word_frequencies,
                                                           class_frequencies=class_frequencies)

    # Compute P(email | ham) with the function you defined just above
    log_prob_email_given_ham = log_prob_email_given_class(treated_emails, cls='ham', word_frequencies=word_frequencies,
                                                          class_frequencies=class_frequencies)

    # Compute P(spam) using the class_frequency dictionary and using the formula #spam emails / #total emails
    p_spam = class_frequencies['spam'] / (class_frequencies['ham'] + class_frequencies['spam'])

    # Compute P(ham) using the class_frequency dictionary and using the formula #ham emails / #total emails
    p_ham = class_frequencies['ham'] / (class_frequencies['ham'] + class_frequencies['spam'])

    # Compute the quantity log(P(spam)) + log(P(email | spam)), let's call it log_spam_likelihood
    log_spam_likelihoods = np.log(p_spam) + log_prob_email_given_spam

    # Compute the quantity P(ham) * P(email | ham), let's call it ham_likelihood
    log_ham_likelihoods = np.log(p_ham) + log_prob_email_given_ham

    # In case of passing return_likelihood = True, then return the desired tuple
    if return_likelihood :
        return log_spam_likelihoods, log_ham_likelihoods

    # Compares both values and choose the class corresponding to the higher value.
    # As the logarithm is an increasing function, the class with the higher value retains this property.
    if log_spam_likelihoods >= log_ham_likelihoods :
        return 1
    else :
        return 0


log_spam_likelihood, log_ham_likelihood = log_naive_bayes(treated_email, word_frequencies= word_frequency, class_frequencies= class_frequency, return_likelihood = True)
print(f"log_spam_likelihood: {log_spam_likelihood}\nlog_ham_likelihood: {log_ham_likelihood}")

print(f"The example email is labeled as: {Y[example_index]}")
print(f"Log Naive bayes model classifies it as: {log_naive_bayes(treated_email, word_frequencies= word_frequency, class_frequencies= class_frequency)}")

# Let's get the predictions for the test set:

# Create an empty list to store the predictions
Y_pred = []


# Iterate over every email in the test set
for email in X_test:
    # Perform prediction
    prediction = log_naive_bayes(email,word_frequencies = word_frequency, class_frequencies = class_frequency)
    # Add it to the list
    Y_pred.append(prediction)

# Get the number of true positives:
true_positives = get_true_positives(Y_test, Y_pred)

# Get the number of true negatives:
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

# Compute the accuracy by summing true negatives with true positives and dividing it by the total number of elements in the dataset.
# Since both Y_pred and Y_test have the same length, it does not matter which one you use.
accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"The accuracy is: {accuracy:.4f}")