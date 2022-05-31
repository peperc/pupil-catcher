import os
import logging
import pandas as pd

from weka.core.converters import Saver, Loader
from weka.filters import Filter
from weka.core import jvm



HEADER = [
    '@relation pressed_keys\n\n',
    '@attribute pressed_key numeric\n',
    '@attribute left_x numeric\n',
    '@attribute left_y numeric\n',
    '@attribute right_x numeric\n',
    '@attribute right_y numeric\n',
    '@attribute centre_x numeric\n',
    '@attribute centre_y numeric\n',
    '@attribute head_angle numeric\n\n',
    '@data\n'
]

def get_test_from_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path, header=0, engine='openpyxl')


def preprocess(keystroke: str) -> str:
    # Noise first
    parameters = keystroke.split(',')
    if parameters[0] in [str(12), str(25), str(127)]: return None

    # Changes to match the key
    if parameters[0] == str(33): return str(49) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(34): return str(50) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(40): return str(56) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(41): return str(57) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(47): return str(55) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(58): return str(46) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(59): return str(44) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(62): return str(60) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(63): return str(39) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(168): return str(180) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(191): return str(161) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(193): return str(97) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(201): return str(101) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(205): return str(105) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(211): return str(111) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(218): return str(117) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(225): return str(97) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(233): return str(101) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(237): return str(105) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(243): return str(111) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(250): return str(117) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(199): return str(231) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(242): return str(111) + keystroke[len(parameters[0]):]
    elif parameters[0] == str(95): return str(45) + keystroke[len(parameters[0]):]

    else: return keystroke
    

def join_and_preprocess_tests(file_names: list[str], output_path: str):
    o = open(output_path, 'x')
    o.writelines(HEADER)

    for file_name in file_names:
        with open('./datasets/' + file_name + '.arff', 'r') as test:
            keystrokes = test.readlines()
            for keystroke in keystrokes[12:]:
                keystroke = preprocess(keystroke)
                if keystroke: o.write(keystroke)

    o.close()

    # Start JVM
    jvm.logger.setLevel(logging.ERROR)
    jvm.start(packages=True)
    
    # Loads it
    loader = Loader(classname="weka.core.converters.ArffLoader")
    dataset = loader.load_file(output_path)

    # Make pressed_key nominal
    filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R","1"])
    filter.inputformat(dataset)
    dataset = filter.filter(dataset)

    # Save the dataset
    os.remove(output_path) 
    saver = Saver(classname="weka.core.converters.ArffSaver")
    saver.save_file(dataset, output_path)

    jvm.stop()



if __name__ == '__main__':
    excel_path = './datasets/arff_index.xlsx'
    dataset_path = './all_together.arff'
    tests = get_test_from_excel(excel_path)
    join_and_preprocess_tests(tests['File name'], dataset_path)

