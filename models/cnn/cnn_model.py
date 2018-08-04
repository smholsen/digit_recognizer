import numpy as np
import tensorflow as tf
import models.data.mnist_data as data
from PIL import Image
import uuid
import os
import base64


class Model(object):
    variable_storage = 'models/cnn/stored_tf_variables/saved_cnn_model.ckpt'
    variable_storage2 = 'models/cnn/stored_tf_variables/saved_cnn_model_further_trained.ckpt'

    def __init__(self):
        return

    def build_network(self):
        # Reset Graph
        tf.reset_default_graph()
        # Each image has 784 pixels (inputs)
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_classes = tf.placeholder(tf.float32, shape=[None, 10])

        # Initialize weights with minor noise for symmetry breaking
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        # Initialize biases with minor positivity to avoid dead neurons
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # Initialize convolution with with stride size of 1
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        # Initialize our pooling as max pooling over 2x2 blocks.
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        # First convolutional layer with 32 features, one for each 5x5 patch. 1 is input channel and
        # 32 is the number of output channels.
        W_conv1 = weight_variable([5, 5, 1, 32])
        # We also include a bias vector with one component for each output channel
        b_conv1 = bias_variable([32])

        # Each image is reshaped to a 4d tensor of 28 by 28 where the final dimension is the number of color channels
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # Add bias, apply ReLU and then max pool. h_pool1 will then be of size 14x14
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Deepen with another layer, this time with 64 features for each 5x5 patch.
        # h_pool2 will then be of size 7x7
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Add a fully connected layer with 1024 neurons for processing entire image.
        # Reshae tensor from pooling layer into a batch of vectors, multiply by weight matrix add bias and then ReLU
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # We add Dropout to avoid overfitting. It's really cool
        # https://www.youtube.com/watch?v=NhZVe50QwPM
        # First we add a placeholder for the probability of keeping the value of a neurons outpur
        keep_prob = tf.placeholder(tf.float32)
        # Then add the dropout, we'll turn it off during testing
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Final layer
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Initialize loss function
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_classes, logits=y_conv))
        # Initialize optimization algorithm ADAM.
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # Define a correct prediction to occur when the prediction is equal to the label
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_classes, 1))

        return {
            'x': x,
            'y_classes': y_classes,
            'W_conv1': W_conv1,
            'b_conv1': b_conv1,
            'x_image': x_image,
            'h_conv1': h_conv1,
            'h_pool1': h_pool1,
            'W_conv2': W_conv2,
            'b_conv2': b_conv2,
            'h_conv2': h_conv2,
            'h_pool2': h_pool2,
            'W_fc1': W_fc1,
            'b_fc1': b_fc1,
            'h_pool2_flat': h_pool2_flat,
            'h_fc1': h_fc1,
            'keep_prob': keep_prob,
            'h_fc1_drop': h_fc1_drop,
            'W_fc2': W_fc2,
            'b_fc2': b_fc2,
            'y_conv': y_conv,
            'cross_entropy': cross_entropy,
            'train_step': train_step,
            'correct_prediction': correct_prediction,
        }

    def initial_train(self):
        print('starting initial train')
        learn = tf.contrib.learn
        tf.logging.set_verbosity(tf.logging.ERROR)

        # Data
        train, validation, test = data.get_sets(5000)

        network = self.build_network()
        y_classes = network['y_classes']
        x = network['x']
        keep_prob = network['keep_prob']
        correct_prediction = network['correct_prediction']
        train_step = network['train_step']

        # Define the accuracy to be the mean of the bools in the output, where the bools are
        # casted to 0's and 1's.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initialize Saver to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            # Run the training in batches
            for i in range(20000):
                batch = train.get_next(50)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_classes: batch[1], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                # Run training with a probability of keeping output in the dropout of 0.5
                train_step.run(feed_dict={x: batch[0], y_classes: batch[1], keep_prob: 0.5})
            # Print results and keep all output during testing
            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: test.images, y_classes: test.labels, keep_prob: 1.0}))
            saver.save(sess, self.variable_storage)

    def predict(self, image):
        img_array = self.prepare_image(image)
        if img_array is None:
            return 'Empty'
        # Build the network before we restore the variables from disk
        network = self.build_network()
        x = network['x']
        keep_prob = network['keep_prob']
        y_conv = network['y_conv']
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Use further trained vriables
            saver.restore(sess, self.variable_storage2)
            feed_dict = {x: [img_array], keep_prob: 1.0}
            classification = sess.run(y_conv, feed_dict)
            return np.argmax(classification[0])

    def train_on_single_img(self, image, label):
        img_array = self.prepare_image(image)
        if img_array is None:
            return
        # Build the network before we restore the variables from disk, then we'll train and save the new vars.
        network = self.build_network()
        train_step = network['train_step']
        x = network['x']
        y_classes = network['y_classes']
        keep_prob = network['keep_prob']
        saver = tf.train.Saver()

        # Convert label to one-hot
        one_hot = np.zeros(10)
        one_hot[label] = 1

        with tf.Session() as sess:
            # Restore and save further trained variables.
            saver.restore(sess, self.variable_storage2)
            feed_dict = {x: [img_array], y_classes: [one_hot], keep_prob: 1}
            train_step.run(feed_dict=feed_dict)
            saver.save(sess, self.variable_storage2)
            return

    def prepare_image(self, image):
        # Save image so that we can process it wil PIL.Image
        # uuid creates a unique identifier
        filename = 'image_' + str(uuid.uuid4()) + '.jpg'
        with open('tmp/' + filename, 'wb') as f:
            f.write(image)

        img = Image.open('tmp/' + filename)

        # Get the coordinates of the digit without padding
        bbox = img.getbbox()
        if bbox is None:
            return None
        # Get width and height of bounding box
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        # But we need it to fit in our 28x28 final img, so we are going to resize it
        if height > width:
            width = int(20.0 * width / height)
            height = 20
        else:
            height = int(20.0 * width / height)
            width = 20

        # We are going to place the cropped digit into the center of our 28x28,
        # So now we must find the x and y coords of the top left where we should
        # paste the image (divide by 2 for mirror padding
        y = int((28 - height) / 2)
        x = int((28 - width) / 2)
        cropped = img.crop(bbox).resize((width, height))
        # Create new image of size 28x28
        new_img = Image.new('L', (28, 28), 255)
        # Paste the digit in the center of the new image
        # sinc our image i sof size 4x4, and total size is 28, there are 3 px to be padded
        new_img.paste(cropped, (x, y), cropped)
        # Extract the data from the image into a numpy array
        new_img.show()
        imgdata = list(new_img.getdata())
        # The numbers are inverted from the initial training data, so we just invert these back
        img_array = np.array([255 - x for x in imgdata])
        # Delete the image file we created earlier
        os.remove('tmp/' + filename)
        return img_array
