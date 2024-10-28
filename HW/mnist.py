import numpy as np

def load_data():
    train_labels_path = 'train-labels-idx1-ubyte'
    train_images_path = 'train-images-idx3-ubyte'
    test_labels_path = 't10k-labels-idx1-ubyte'
    test_images_path = 't10k-images-idx3-ubyte'

    with open(train_labels_path, 'rb') as lbpath:
        lbpath.read(8)
        train_labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(train_images_path, 'rb') as imgpath:
        imgpath.read(16)
        train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 28, 28)

    with open(test_labels_path, 'rb') as lbpath:
        lbpath.read(8)
        test_labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(test_images_path, 'rb') as imgpath:
        imgpath.read(16)
        test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 28, 28)

    return (train_images, train_labels), (test_images, test_labels)
