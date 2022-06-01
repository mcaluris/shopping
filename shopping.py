import csv
import sys
import random
from csv import reader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from format import *
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity, total = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"Accuracy: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    key = ["int", "float", "int", "float", "int", "float", "float", "float",
           "float", "float", "month", "int", "int", "int", "int", {"New_Visitor": 0, "Returning_Visitor": 1, "Other": 0}, "boolean"]

    with open(filename, 'r') as f:
        csv_reader = reader(f)
        next(csv_reader)

        data = []

        for row in csv_reader:
            data.append({
                "evidence": [cell for cell in row[:17]],
                "label": 1 if row[17] == "TRUE" else 0
            })

    for row in data:
        for index, variable in enumerate(row["evidence"]):
            row["evidence"][index] = Format.format(variable, key[index])

    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    #model = KNeighborsClassifier(n_neighbors=1)
    #model = Perceptron()
    #model = SVC()
    model = GaussianNB()
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).
    """
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    accuracy = accuracy_score(labels, predictions)

    return sensitivity, specificity, accuracy


if __name__ == "__main__":
    main()
