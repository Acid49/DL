import numpy as np
import gzip

def load_data():
    train_labels_path = 'train-labels-idx1-ubyte.gz'
    train_images_path = 'train-images-idx3-ubyte.gz'
    test_labels_path = 't10k-labels-idx1-ubyte.gz'
    test_images_path = 't10k-images-idx3-ubyte.gz'

    with gzip.open(train_labels_path, 'rb') as f:
        f.read(8)
        buf = f.read()
        train_labels = np.frombuffer(buf, dtype=np.uint8)

    with gzip.open(train_images_path, 'rb') as f:
        f.read(16)
        buf = f.read()
        train_images = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)

    with gzip.open(test_labels_path, 'rb') as f:
        f.read(8)
        buf = f.read()
        test_labels = np.frombuffer(buf, dtype=np.uint8)

    with gzip.open(test_images_path, 'rb') as f:
        f.read(16)
        buf = f.read()
        test_images = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 28, 28)

    return (train_images, train_labels), (test_images, test_labels)
