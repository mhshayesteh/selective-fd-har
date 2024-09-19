from numpy import loadtxt
import numpy as np
from tqdm import tqdm


def txt2float(path_file):
    txt_file = open(path_file, 'r')
    lines = txt_file.readlines()
    final_result = []
    for line in lines:
        values = line.split(' ')
        arr_float = []
        for v in values:
            if v != '':
                arr_float.append(float(v))
        final_result.append(arr_float)
    return final_result


def create_2d_array_image(signals):
    # Stack all signals vertically to create a 2D array
    image_2d = np.vstack([signals['body_acc_x'], signals['body_acc_y'], signals['body_acc_z'],
                          signals['body_gyro_x'], signals['body_gyro_y'], signals['body_gyro_z'],
                          signals['total_acc_x'], signals['total_acc_y'], signals['total_acc_z']])

    return image_2d


def prepare_handcrafted_dataset():
    root_path = 'data/uci-har'
    train_path = root_path + '/train'
    test_path = root_path + '/test'

    x_train = txt2float(train_path + '/X_train.txt')
    y_train = loadtxt(train_path + '/y_train.txt', comments="#", delimiter=" ", unpack=False)
    s_train = loadtxt(train_path + '/subject_train.txt', comments="#", delimiter=" ", unpack=False)

    x_test = txt2float(test_path + '/X_test.txt')
    y_test = loadtxt(test_path + '/y_test.txt', comments="#", delimiter=" ", unpack=False)
    s_test = loadtxt(test_path + '/subject_test.txt', comments="#", delimiter=" ", unpack=False)

    x_hc = np.array(x_train + x_test)
    y_hc = np.concatenate((y_train, y_test))
    s_hc = np.concatenate((s_train, s_test))

    return x_hc, y_hc, s_hc


def prepare_raw_data_dataset():
    root_path = 'data/uci-har'
    train_path = root_path + '/train'
    test_path = root_path + '/test'

    train_inertial_path = train_path + '/Inertial Signals'
    test_inertial_path = test_path + '/Inertial Signals'

    y_train = np.loadtxt(train_path + '/y_train.txt', comments="#", delimiter=" ", unpack=False)
    y_test = np.loadtxt(test_path + '/y_test.txt', comments="#", delimiter=" ", unpack=False)

    s_train = np.loadtxt(train_path + '/subject_train.txt', comments="#", delimiter=" ", unpack=False)
    s_test = np.loadtxt(test_path + '/subject_test.txt', comments="#", delimiter=" ", unpack=False)

    body_acc_x_train = np.loadtxt(train_inertial_path + '/body_acc_x_train.txt')
    body_acc_y_train = np.loadtxt(train_inertial_path + '/body_acc_y_train.txt')
    body_acc_z_train = np.loadtxt(train_inertial_path + '/body_acc_z_train.txt')
    body_gyro_x_train = np.loadtxt(train_inertial_path + '/body_gyro_x_train.txt')
    body_gyro_y_train = np.loadtxt(train_inertial_path + '/body_gyro_y_train.txt')
    body_gyro_z_train = np.loadtxt(train_inertial_path + '/body_gyro_z_train.txt')
    total_acc_x_train = np.loadtxt(train_inertial_path + '/total_acc_x_train.txt')
    total_acc_y_train = np.loadtxt(train_inertial_path + '/total_acc_y_train.txt')
    total_acc_z_train = np.loadtxt(train_inertial_path + '/total_acc_z_train.txt')

    body_acc_x_test = np.loadtxt(test_inertial_path + '/body_acc_x_test.txt')
    body_acc_y_test = np.loadtxt(test_inertial_path + '/body_acc_y_test.txt')
    body_acc_z_test = np.loadtxt(test_inertial_path + '/body_acc_z_test.txt')
    body_gyro_x_test = np.loadtxt(test_inertial_path + '/body_gyro_x_test.txt')
    body_gyro_y_test = np.loadtxt(test_inertial_path + '/body_gyro_y_test.txt')
    body_gyro_z_test = np.loadtxt(test_inertial_path + '/body_gyro_z_test.txt')
    total_acc_x_test = np.loadtxt(test_inertial_path + '/total_acc_x_test.txt')
    total_acc_y_test = np.loadtxt(test_inertial_path + '/total_acc_y_test.txt')
    total_acc_z_test = np.loadtxt(test_inertial_path + '/total_acc_z_test.txt')

    # Concatenate training and test data
    body_acc_x = np.concatenate((body_acc_x_train, body_acc_x_test))
    body_acc_y = np.concatenate((body_acc_y_train, body_acc_y_test))
    body_acc_z = np.concatenate((body_acc_z_train, body_acc_z_test))
    body_gyro_x = np.concatenate((body_gyro_x_train, body_gyro_x_test))
    body_gyro_y = np.concatenate((body_gyro_y_train, body_gyro_y_test))
    body_gyro_z = np.concatenate((body_gyro_z_train, body_gyro_z_test))
    total_acc_x = np.concatenate((total_acc_x_train, total_acc_x_test))
    total_acc_y = np.concatenate((total_acc_y_train, total_acc_y_test))
    total_acc_z = np.concatenate((total_acc_z_train, total_acc_z_test))

    y_combined = np.concatenate((y_train, y_test))
    subjects_combined = np.concatenate((s_train, s_test))

    # Initialize the 2D image data list
    images_2d = []

    for i in tqdm(range(len(body_acc_x))):
        signals = {
            'body_acc_x': body_acc_x[i],
            'body_acc_y': body_acc_y[i],
            'body_acc_z': body_acc_z[i],
            'body_gyro_x': body_gyro_x[i],
            'body_gyro_y': body_gyro_y[i],
            'body_gyro_z': body_gyro_z[i],
            'total_acc_x': total_acc_x[i],
            'total_acc_y': total_acc_y[i],
            'total_acc_z': total_acc_z[i]
        }

        # Create 2D array image from all signals
        image_2d = create_2d_array_image(signals)
        images_2d.append(image_2d)

    return np.array(images_2d), y_combined, subjects_combined
