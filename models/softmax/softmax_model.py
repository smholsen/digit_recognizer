import tensorflow as tf
import models.data.mnist_data as data

learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# Data
train, validation, test = data.get_sets(5000)

sess = tf.InteractiveSession()

# Each image has 784 pixels (inputs)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

weights = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

# Iitialize the variables
sess.run(tf.global_variables_initializer())

y = tf.matmul(x, weights) + bias

# Loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# Gradient Descent function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Repeatedly run gradient descent on the model to train it
# We replace the input after every 100 image
for _ in range(1000):
    batch = train.get_next(100)
    train_step.run(feed_dict={x: batch[0], y_true: batch[1]})

# If output of model is equal to the label then we classify it as a correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
# To get accuracy we cast the bools to floats and take the mean value from the list. (e.g. [1, 0, 1, 1} = 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test on testing set
print(accuracy.eval(feed_dict={x: test.images, y_true: test.labels}))

# ~.85