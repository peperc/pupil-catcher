import os
import logging
import pandas as pd

from weka.core.converters import Saver, Loader
from weka.filters import Filter
from weka.core import jvm


HEADER = [
    '@relation pressed_keys\n\n',
    '@attribute left_x numeric\n',
    '@attribute left_y numeric\n',
    '@attribute right_x numeric\n',
    '@attribute right_y numeric\n',
    '@attribute centre_x numeric\n',
    '@attribute centre_y numeric\n',
    '@attribute head_angle numeric\n',
    '@attribute pressed_key numeric\n\n',
    '@data\n'
]

def get_test_from_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path, header=0, engine='openpyxl')


def filter_tests(tests: pd.DataFrame, gender: str = None, old: str = None, glasses: str = None, forced: str = None, lighting: int = None) -> pd.DataFrame:
    # man or woman
    if gender:
        tests = tests[tests['Gender'] == gender]
    
    # yes or no
    if old:
        tests = tests[tests['Old'] == old]
    
    # yes or no
    if glasses:
        tests = tests[tests['Glasses'] == glasses]

    # yes or no
    if forced:
        tests = tests[tests['Forced'] == forced]

    # 1, 2 or 3
    if lighting:
        tests = tests[tests['Lighting'] == lighting]
    
    return tests



def preprocess(keystroke: str) -> str:
    # Noise first
    parameters = keystroke.split(',')
    if parameters[0] in [str(12), str(25), str(127), str(32)]: return None

    # Changes to match the key
    if parameters[0] == str(33): return keystroke[len(parameters[0])+1:-1] + ',' + str(49) + '\n'
    elif parameters[0] == str(34): return keystroke[len(parameters[0])+1:-1] + ',' + str(50) + '\n'
    elif parameters[0] == str(40): return keystroke[len(parameters[0])+1:-1] + ',' + str(56) + '\n'
    elif parameters[0] == str(41): return keystroke[len(parameters[0])+1:-1] + ',' + str(57) + '\n'
    elif parameters[0] == str(47): return keystroke[len(parameters[0])+1:-1] + ',' + str(55) + '\n'
    elif parameters[0] == str(58): return keystroke[len(parameters[0])+1:-1] + ',' + str(46) + '\n'
    elif parameters[0] == str(59): return keystroke[len(parameters[0])+1:-1] + ',' + str(44) + '\n'
    elif parameters[0] == str(62): return keystroke[len(parameters[0])+1:-1] + ',' + str(60) + '\n'
    elif parameters[0] == str(63): return keystroke[len(parameters[0])+1:-1] + ',' + str(39) + '\n'
    elif parameters[0] == str(168): return keystroke[len(parameters[0])+1:-1] + ',' + str(180) + '\n'
    elif parameters[0] == str(191): return keystroke[len(parameters[0])+1:-1] + ',' + str(161) + '\n'
    elif parameters[0] == str(193): return keystroke[len(parameters[0])+1:-1] + ',' + str(97) + '\n'
    elif parameters[0] == str(201): return keystroke[len(parameters[0])+1:-1] + ',' + str(101) + '\n'
    elif parameters[0] == str(205): return keystroke[len(parameters[0])+1:-1] + ',' + str(105) + '\n'
    elif parameters[0] == str(211): return keystroke[len(parameters[0])+1:-1] + ',' + str(111) + '\n'
    elif parameters[0] == str(218): return keystroke[len(parameters[0])+1:-1] + ',' + str(117) + '\n'
    elif parameters[0] == str(225): return keystroke[len(parameters[0])+1:-1] + ',' + str(97) + '\n'
    elif parameters[0] == str(233): return keystroke[len(parameters[0])+1:-1] + ',' + str(101) + '\n'
    elif parameters[0] == str(237): return keystroke[len(parameters[0])+1:-1] + ',' + str(105) + '\n'
    elif parameters[0] == str(243): return keystroke[len(parameters[0])+1:-1] + ',' + str(111) + '\n'
    elif parameters[0] == str(250): return keystroke[len(parameters[0])+1:-1] + ',' + str(117) + '\n'
    elif parameters[0] == str(199): return keystroke[len(parameters[0])+1:-1] + ',' + str(231) + '\n'
    elif parameters[0] == str(242): return keystroke[len(parameters[0])+1:-1] + ',' + str(111) + '\n'
    elif parameters[0] == str(95): return keystroke[len(parameters[0])+1:-1] + ',' + str(45) + '\n'

    else: return keystroke[len(parameters[0])+1:-1] + ',' + parameters[0] + '\n'
    

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
    filter = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R","last"])
    filter.inputformat(dataset)
    dataset = filter.filter(dataset)

    # Save the dataset
    os.remove(output_path) 
    saver = Saver(classname="weka.core.converters.ArffSaver")
    saver.save_file(dataset, output_path)

    jvm.stop()



if __name__ == '__main__':
    excel_path = './datasets/arff_index.xlsx'
    output_path = './datasets/aggregated/11_light_3_no_spaces.arff'
    tests = get_test_from_excel(excel_path)

    # Filter!
    tests = filter_tests(tests, lighting=3)

    join_and_preprocess_tests(tests['File name'], output_path)
