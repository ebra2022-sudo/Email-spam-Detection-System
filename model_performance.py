from email_spam_detection_system import X_test, word_frequency, class_frequency, Y_test, preprocess_text
from naive_bayes_algorithim import naive_bayes


def get_true_positives(Y_true, Y_pred) :
    """
    Calculate the number of true positive instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of true positives, where true label and predicted label are both 1.
    """
    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred) :
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_positives = 0
    # Iterate over the number of elements in the list
    for i in range(n) :
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 1 and predicted_label_i = 1 (true positives)
        if true_label_i == 1 and predicted_label_i == 1 :
            true_positives += 1
    return true_positives


def get_true_negatives(Y_true, Y_pred) :
    """
    Calculate the number of true negative instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of true negatives, where true label and predicted label are both 0.
    """

    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred) :
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_negatives = 0
    # Iterate over the number of elements in the list
    for i in range(n) :
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 0 and predicted_label_i = 0 (true negatives)
        if true_label_i == 0 and predicted_label_i == 0 :
            true_negatives += 1
    return true_negatives

# Let's get the predictions for the test set:

# Create an empty list to store the predictions
Y_pred = []


# Iterate over every email in the test set
for email in X_test:
    # Perform prediction
    prediction = naive_bayes(email, word_frequency, class_frequency)
    # Add it to the list
    Y_pred.append(prediction)

# Checking if both Y_pred and Y_test (these are the true labels) match in length:
print(f"Y_test and Y_pred matches in length? Answer: {len(Y_pred) == len(Y_test)}")


# Get the number of true positives:
true_positives = get_true_positives(Y_test, Y_pred)

# Get the number of true negatives:
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

# Compute the accuracy by summing true negatives with true positives and dividing it by the total number of elements in the dataset.
# Since both Y_pred and Y_test have the same length, it does not matter which one you use.
accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"Accuracy is: {accuracy:.4f}")

email = "Please meet me in 2 hours in the main building. I have an important task for you."
# email = "You win a lottery prize! Congratulations! Click here to claim it"

# Preprocess the email
treated_email = preprocess_text(email)
# Get the prediction, in order to print it nicely, if the output is 1 then the prediction will be written as "spam" otherwise "ham".
prediction = "spam" if naive_bayes(treated_email, word_frequency, class_frequency) == 1 else "ham"
print(f"The email is: {email}\nThe model predicts it as {prediction}.")