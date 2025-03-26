"""
Hidden problem in the Naive Bayes model.
A hidden problem in the current model is impacting its performance.
Let's delve into the issue by manually performing the Naive Bayes computation on a specific example.
"""


"""
The Underflow Problem
The challenge you encounter is termed an underflow problem, indicating that you are dealing with exceedingly 
small numbers beyond the computer's precision. In this case, the root cause is the very large product involved 
in Naive Bayes calculations. Fortunately, there is a solution to this issue. use p(email| class) = log(πp(word i| class)
"""
from email_spam_detection_system import X, preprocess_text, word_frequency, class_frequency, Y
from naive_bayes_algorithim import naive_bayes

example_index = 4798
example_email = X[example_index]
treated_email = preprocess_text(example_email)
print(f"The email is:\n\t{example_email}\n\nAfter preprocessing:\n\t:{treated_email}")

spam_likelihood, ham_likelihood = naive_bayes(treated_email, word_frequency = word_frequency, class_frequency = class_frequency, return_likelihood = True)
print(f"spam_likelihood: {spam_likelihood}\nham_likelihood: {ham_likelihood}")

print(f"The example email is labeled as: {Y[example_index]}")
print(f"Naive bayes model classifies it as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

"""
spam_likelihood: 0.0
ham_likelihood: 0.0
This is weird, both spam and ham likelihood are 0
! How can it be possible? By the way, by the actual rule, the model classifies as 1 (spam) if spam\_likelihood≥ham\_likelihood
, so this email would be classified as spam. Let's compare the true and predicted labels."""