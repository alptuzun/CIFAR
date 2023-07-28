import datetime
import random
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from pgd_attack import LinfPGDAttack
from model import Model
import matplotlib.pyplot as plt

def train_step(x_batch, y_batch, model, optimizer):
    with tf.GradientTape() as tape:
        nat_logits = model(x_batch, training=True)
        nat_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch, logits=nat_logits)  # Corrected argument order
        loss = tf.reduce_mean(nat_loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    nat_accuracy = model.calculate_accuracy(x_batch, y_batch)

    return loss, nat_accuracy

random_seed = 4557077
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

max_num_training_steps = 300
num_output_steps = 10
num_summary_steps = 10
num_checkpoint_steps = 300
batch_size = 50

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
model = Model()
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)  # Use TF 2.x Adam optimizer

model_dir = "models/not_robust_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

summary_writer = tf.summary.create_file_writer(model_dir)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)

training_accuracy = 0.0

training_time = datetime.timedelta(seconds=0)

for ii, (x_batch, y_batch) in enumerate(train_dataset):
    start_time = datetime.datetime.now()

    # Tensorboard summaries
    if ii % num_summary_steps == 0:
        nat_accuracy = model.calculate_accuracy(x_batch, y_batch)

        with tf.summary.create_file_writer(model_dir).as_default():
            tf.summary.scalar('accuracy nat train', nat_accuracy, step=global_step)
            x_image = tf.reshape(x_batch, [-1, 28, 28, 1])
            tf.summary.image('images nat train', x_image, step=global_step)

    # Output to stdout
    if ii % num_output_steps == 0:
        print('Step {}:    ({})'.format(ii, datetime.datetime.now()))
        print('    training nat accuracy {:.4}%'.format(training_accuracy * 100))
        if ii != 0:
            num_examples_per_sec = num_output_steps * batch_size / training_time.total_seconds()
            print('    {} examples per second'.format(num_examples_per_sec))
        training_time = datetime.timedelta(seconds=0)
    
    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
        checkpoint_path = os.path.join(model_dir, 'checkpoint')
        model.save_weights(checkpoint_path)

    # Actual training step
    loss, training_accuracy = train_step(x_batch, y_batch, model, optimizer)

    end_time = datetime.datetime.now()
    training_time += end_time - start_time

    if ii >= max_num_training_steps:
        break

# After training, let's evaluate the model on the test set
test_accuracy = model.calculate_accuracy(x_test, y_test)
print('Test Accuracy: {:.4}%'.format(test_accuracy * 100))

epsilon = 0.9  # Perturbation budget for adversarial examples
num_adv_steps = 10  # Number of PGD steps to generate adversarial examples
adv_step_size = 0.08  # Step size for each PGD step

# Step 2: Adversarial Training
num_adv_examples = int(len(x_train) * 0.5)  # Number of adversarial examples to generate per epoch

num_epochs = 2

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    for ii, (x_batch, y_batch) in enumerate(train_dataset):
        # Generate adversarial examples for this batch
        pgd_attack = LinfPGDAttack(model, epsilon, num_adv_steps, adv_step_size, random_start=True,
                                   loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        x_adv_batch = pgd_attack.perturb(x_batch, y_batch)

        # Combine original and adversarial examples
        combined_x_batch = tf.concat([x_batch, x_adv_batch], axis=0)
        combined_y_batch = tf.concat([y_batch, y_batch], axis=0)

        # Shuffle the combined data
        combined_dataset = tf.data.Dataset.from_tensor_slices((combined_x_batch, combined_y_batch))
        combined_dataset = combined_dataset.shuffle(buffer_size=len(combined_x_batch)).batch(batch_size)

        # Train the model on the combined and shuffled data
        for (x_train_batch, y_train_batch) in combined_dataset:
            loss, training_accuracy = train_step(x_train_batch, y_train_batch, model, optimizer)
            end_time = datetime.datetime.now()
            training_time += end_time - start_time

        # Output training progress
        if ii % num_output_steps == 0:
            print('Step {}:    ({})'.format(ii, datetime.datetime.now()))
            print('    training adv accuracy {:.4}%'.format(training_accuracy * 100))
            if ii != 0:
                num_examples_per_sec = num_output_steps * batch_size / training_time.total_seconds()
                print('    {} examples per second'.format(num_examples_per_sec))
            training_time = datetime.timedelta(seconds=0)
        
        

    # Write a checkpoint at the end of each epoch
    checkpoint_path = os.path.join(model_dir, 'checkpoint_epoch_{}'.format(epoch))
    model.save_weights(checkpoint_path)

# After adversarial training, let's evaluate the model on the test set
test_accuracy = model.calculate_accuracy(x_test, y_test)
print('Adversarial Test Accuracy: {:.4}%'.format(test_accuracy * 100))