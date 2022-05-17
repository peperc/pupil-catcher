import logging
import os
import time

import weka.core.jvm as jvm

from lib.weka_classifier import get_prediction, load_classifier, load_dataset, train_classifier, save_classifier


# Start JVM
jvm.logger.setLevel(logging.ERROR)
jvm.start(packages=True)

# Load the train dataset
dataset = load_dataset(os.path.join(os.path.dirname(__file__), 'datasets/2022-05-17_13-41-02.arff'))

# Train a classifier
# classifier = train_classifier(dataset)

# Save it
# save_classifier(classifier, os.path.join(os.path.dirname(__file__), f'models/{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.model'))

# Load the classifier
classifier, _ = load_classifier(os.path.join(os.path.dirname(__file__), f'models/2022-05-17_13-44-24.model'))

# Load the evaluation dataset
# dataset = load_dataset(os.path.join(os.path.dirname(__file__), 'datasets/2022-05-17_13-27-33.arff'))

# Predict text
file = open("predicted.txt", "w")
for inst in dataset:
    file.write(get_prediction(classifier, dataset, inst))
file.close

jvm.stop()

