def create_dataset():
    return [
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


def add_to_dataset(dataset: list[str], key, left_eye, right_eye, centre, angle):
    values = ''
    if key: values += str(key) + ','
    else: values += '?,'
    if left_eye: values += str(left_eye[0]) + ',' + str(left_eye[1]) + ','
    else: values += '?,?,'
    if right_eye: values += str(right_eye[0]) + ',' + str(right_eye[1]) + ','
    else: values += '?,?,'
    values += str(centre[0]) + ',' + str(centre[1]) + ',' + str(angle) + '\n'
    
    dataset.append(values)


def save_dataset(dataset: list[str], path: str):
    # The filter for making NumericToNominal is not available :(

    # Save the dataset
    with open(path, 'a') as f:
        f.writelines(dataset)
