from weka.core.dataset import Attribute, Instances, Instance
from weka.core.converters import Saver, Loader
from weka.filters import Filter
from weka.classifiers import Classifier


def create_dataset():
    pressed_key = Attribute.create_numeric("pressed_key")
    left_x = Attribute.create_numeric("left_x")
    left_y = Attribute.create_numeric("left_y")
    right_x = Attribute.create_numeric("right_x")
    right_y = Attribute.create_numeric("right_y")
    centre_x = Attribute.create_numeric("centre_x")
    centre_y = Attribute.create_numeric("centre_y")
    head_angle = Attribute.create_numeric("head_angle")

    return Instances.create_instances("pressed_keys", [pressed_key, left_x, left_y, right_x, right_y, centre_x, centre_y, head_angle], 0)


def add_to_dataset(dataset, key, left_eye, right_eye, centre, angle):
    values = []
    if key: values.append(key)
    else: values.append(Instance.missing_value())
    if left_eye: values.extend([left_eye[0], left_eye[1]])
    else: values.extend([Instance.missing_value(), Instance.missing_value()])
    if right_eye: values.extend([right_eye[0], right_eye[1]])
    else: values.extend([Instance.missing_value(), Instance.missing_value()])
    values.extend([centre[0], centre[1], angle])
    
    inst = Instance.create_instance(values)
    dataset.add_instance(inst)


def save_dataset(dataset, path):
    # First change the first attribute type to nominal
    filter = Filter (classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R","1"])
    filter.inputformat(dataset)
    dataset = filter.filter(dataset)

    # Save the dataset
    saver = Saver(classname="weka.core.converters.ArffSaver")
    saver.save_file(dataset, path)


def load_dataset(path: str) -> Instances:
    # Load the training data
    loader = Loader(classname="weka.core.converters.ArffLoader")
    return loader.load_file(path, class_index='first')


def train_classifier(dataset: Instances) -> Classifier:
    # Train the classifier
    classifier = Classifier(classname="weka.classifiers.trees.J48", options= ["-C", "0.25", "-M", "2"])
    classifier.build_classifier(dataset)
    return classifier


def save_classifier(classifier: Classifier, path: str):
    classifier.serialize(path)


def get_prediction(classifier: Classifier, dataset: Instances, inst: Instance) -> str:
    pred = classifier.classify_instance(inst)
    key = int(dataset.attribute_by_name('pressed_key').value(int(pred)))
    return chr(key)


def load_classifier(path: str) -> Classifier:
    return Classifier.deserialize(path)