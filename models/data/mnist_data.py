import gzip
import numpy as np
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# Constants
VALIDATION_CHECK_IMAGES = 2051
VALIDATION_CHECK_LABELS = 2049


class DataSet(object):
    def __init__(self, images, labels):
        # Convert shape of images from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_images = images.shape[0]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_images(self):
        return self.num_images

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def get_next(self, batch_size):
        start = self._index_in_epoch
        # Shuffle first epoch
        if self._epochs_completed == 0 and start == 0:
            perm0 = np.arange(self._num_images)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_images:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_images - start
            images_rest_part = self._images[start:self._num_images]
            labels_rest_part = self._labels[start:self._num_images]
            # Shuffle the data
            perm = np.arange(self._num_images)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        # Continue Epoch
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def get_sets(validation_size=0):
    TRAIN_IMAGES = 'models/data/train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'models/data/train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 'models/data/t10k-images-idx3-ubyte.gz'
    TEST_LABELS = 'models/data/t10k-labels-idx1-ubyte.gz'

    train_images = extract_images(TRAIN_IMAGES)
    train_labels = extract_labels(TRAIN_LABELS)
    test_images = extract_images(TEST_IMAGES)
    test_labels = extract_labels(TEST_LABELS)

    if validation_size < 0 or validation_size > len(train_images):
        raise ValueError('Invalid size of validation set')

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(images=train_images, labels=train_labels)
    validation = DataSet(images=validation_images, labels=validation_labels)
    test = DataSet(images=test_images, labels=test_labels)
    return train, validation, test


def next_32_bytes(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(images_gzip):
    with gzip.GzipFile(images_gzip) as bytestream:
        validation_check = next_32_bytes(bytestream)
        if not validation_check == VALIDATION_CHECK_IMAGES:
            raise ValueError('Validation of dataset failed (Images)')
        # Number of images in the dataset
        num_images = next_32_bytes(bytestream)
        # Number of rows in each image
        rows = next_32_bytes(bytestream)
        # Number of columns in each image
        cols = next_32_bytes(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        # Each image is formatted as an array of [index, y-axis, x-axis, depth]
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(labels_gzip):
    with gzip.GzipFile(labels_gzip) as bytestream:
        validation_check = next_32_bytes(bytestream)
        if not validation_check == VALIDATION_CHECK_LABELS:
            raise ValueError('Validation of dataset failed (Labels)')
        num_labels = next_32_bytes(bytestream)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        # Convert to one-hot encoding.
        # The 10 comes from there being 10 different values.
        index_offset = np.arange(num_labels) * 10
        labels_one_hot = np.zeros((num_labels, 10))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        return labels_one_hot
