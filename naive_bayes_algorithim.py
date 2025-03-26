from email_spam_detection_system import prob_email_given_class, preprocess_text, word_frequency, class_frequency

"""
 perform both computations below to calculate the probability an email is either spam or ham:

𝑃(spam)⋅𝑃(email∣spam)


𝑃(ham)⋅𝑃(email∣ham)


The one with the greatest value will be the class your algorithm assigns to that email.
Note that the function below includes a parameter that tells the function to return
both probabilities rather than the class that was chosen.

Note: You will notice that the output will be an integer, indicating the respective email class.
It would be possible to return spam if the email is predicted as spam and ham 
if the email is predicted as ham, however, having the model output a number 
helps further computation, such as metrics to evaluate the model performance.
"""

def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood=False) :
    """
    Naive Bayes classifier for spam detection.

    This function calculates the probability of an email being spam (1) or ham (0)
    based on the Naive Bayes algorithm. It uses the conditional probabilities of the
    treated_email given spam and ham, as well as the prior probabilities of spam and ham
    classes. The final decision is made by comparing the calculated probabilities.

    Parameters:
    - treated_email (list): A preprocessed representation of the input email.
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.
        - return_likelihood (bool): If true, it returns the likelihood of both spam and ham.

    Returns:
    If return_likelihood = False:
        - int: 1 if the email is classified as spam, 0 if classified as ham.
    If return_likelihood = True:
        - tuple: A tuple with the format (spam_likelihood, ham_likelihood)
    """

    ### START CODE HERE ###

    # Compute P(email | spam) with the function you defined just above. Don't forget to add the word_frequency and class_frequency parameters!
    prob_email_given_spam = prob_email_given_class(treated_email, cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)

    # Compute P(email | ham) with the function you defined just above. Don't forget to add the word_frequency and class_frequency parameters!
    prob_email_given_ham = prob_email_given_class(treated_email, cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)

    # Compute P(spam) using the class_frequency dictionary and using the formula #spam emails / #total emails
    p_spam = class_frequency["spam"] / len(class_frequency)

    # Compute P(ham) using the class_frequency dictionary and using the formula #ham emails / #total emails
    p_ham = class_frequency["ham"] / len(class_frequency)

    # Compute the quantity P(spam) * P(email | spam), let's call it spam_likelihood
    spam_likelihood = p_spam * prob_email_given_spam

    # Compute the quantity P(ham) * P(email | ham), let's call it ham_likelihood
    ham_likelihood = p_ham * prob_email_given_ham

    ### END CODE HERE ###

    # In case of passing return_likelihood = True, then return the desired tuple
    if return_likelihood == True :
        return (spam_likelihood, ham_likelihood)

    # Compares both values and choose the class corresponding to the higher value
    elif spam_likelihood >= ham_likelihood :
        return 1
    else :
        return 0


example_email = "Click here to win a lottery ticket and claim your prize!"
treated_email = preprocess_text(example_email)

print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

print("\n\n")
example_email = "Our meeting will happen in the main office. Please be there in time."
treated_email = preprocess_text(example_email)

print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {naive_bayes(treated_email, word_frequency, class_frequency)}")