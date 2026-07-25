from sklearn.metrics import accuracy_score, f1_score, recall_score

# Fucntion to calculate metrics. recall_score needs positive label specified for binary classification
# We set average=None to get recall for each class and then first element [0] since we are interested in 
# recall of positive class
def evaluate_model(y_test, predictions, positive_label):
    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(y_test, predictions, average='weighted')
    target_recall = recall_score(
        y_test,
        predictions,
        labels=[positive_label],
        average=None,
        zero_division=0
    )[0]
    return {
        "accuracy" : accuracy,
        "weighted_f1" : weighted_f1,
        "target_recall" : target_recall
    }

# Function to check model upgrade to production promotion
def is_eligible_for_production(accuracy, target_recall, accuracy_threshold, recall_threshold):
    return (accuracy>=accuracy_threshold and target_recall>=recall_threshold)