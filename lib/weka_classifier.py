from numpy import append
from weka.core.dataset import Attribute, Instances, Instance
from weka.core.converters import Saver


def create_dataset():
    pressed_key = Attribute.create_numeric("pressed_key")
    left_x = Attribute.create_numeric("left_x")
    left_y = Attribute.create_numeric("left_y")
    right_x = Attribute.create_numeric("right_x")
    right_y = Attribute.create_numeric("right_y")
    return Instances.create_instances("pressed_keys", [pressed_key, left_x, left_y, right_x, right_y], 0)


def add_to_dataset(dataset, key, left_eye, right_eye):
    values = []
    if key: values.append(key)
    else: values.append(Instance.missing_value())
    if left_eye: values.extend([left_eye[0], left_eye[1]])
    else: values.extend([Instance.missing_value(), Instance.missing_value()])
    if right_eye: values.extend([right_eye[0], right_eye[1]])
    else: values.extend([Instance.missing_value(), Instance.missing_value()])
    
    inst = Instance.create_instance(values)
    dataset.add_instance(inst)


def save_dataset(dataset, path):
    saver = Saver(classname="weka.core.converters.ArffSaver")
    saver.save_file(dataset, path)
