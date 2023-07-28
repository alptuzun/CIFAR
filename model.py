import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x_image = tf.reshape(inputs, [-1, 28, 28, 1])
        h_conv1 = self.pool1(self.conv1(x_image))
        h_conv2 = self.pool2(self.conv2(h_conv1))
        h_flat = self.flatten(h_conv2)
        h_fc1 = self.fc1(h_flat)
        pre_softmax = self.fc2(h_fc1)
        return pre_softmax

    def calculate_accuracy(self, x, y):
        y_pred = tf.argmax(self(x), axis=1)
        y_pred = tf.cast(y_pred, tf.int32)  # Add this line to convert y_pred to int32
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y), tf.float32))
        return accuracy.numpy() if tf.executing_eagerly() else accuracy
