import logging

import weka.core.jvm as jvm

from lib.weka_functions import get_prediction, load_classifier, load_dataset, train_classifier, save_classifier


# Start JVM
jvm.logger.setLevel(logging.ERROR)
jvm.start(packages=True)

# Load the train dataset
# dataset = load_dataset('datasets/2022-05-17_13-41-02.arff')

# Train a classifier
# classifier = train_classifier(dataset)

# Save it
# save_classifier(classifier, f'models/{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.model')

# Load the classifier
classifier, _ = load_classifier('models/svm_all.model')

# Load the evaluation dataset
dataset = load_dataset('datasets/2022-06-11_17-28-50.arff')

# Predict text (this is not working, not getting same results as weka does)
file = open("predicted.txt", "w")
for inst in dataset:
    file.write(get_prediction(classifier, dataset, inst) + '\n')
file.close

jvm.stop()

