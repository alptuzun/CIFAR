import datetime
import os
import sys
import time
import tensorflow as tf
import numpy as np
from pgd_attack import LinfPGDAttack
from model import Model

def train_step_with_adversarial(x_batch, y_batch, model, optimizer, attack):
    with tf.GradientTape() as tape:
        # Clean data loss
        logits = model(x_batch, training=True)
        clean_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch, logits)
        clean_loss = tf.reduce_mean(clean_loss)

        # Adversarial data loss
        x_adv = attack.perturb(x_batch, y_batch)
        adv_logits = model(x_adv, training=True)
        adv_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch, adv_logits)
        adv_loss = tf.reduce_mean(adv_loss)

        # Combine clean and adversarial losses
        total_loss = clean_loss + adv_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def run_attack(checkpoint, x_adv, epsilon):
    model = Model()
    num_eval_examples = 10000

    x_nat = x_adv # x_adv now contains the adversarial examples
    l_inf = np.amax(np.abs(x_nat - x_adv))
  
    if l_inf > epsilon + 0.0001:
        print('maximum perturbation found: {}'.format(l_inf))
        print('maximum perturbation allowed: {}'.format(epsilon))
        return

    y_pred = [] # label accumulator

    # Restore the checkpoint
    model.load_weights(checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(0, num_eval_examples, eval_batch_size):
        x_batch = x_adv[ibatch : ibatch + eval_batch_size]
        y_batch = mnist.test.labels[ibatch : ibatch + eval_batch_size]

        y_pred_batch = model(x_batch, training=False)
        y_pred.append(y_pred_batch)

    y_pred = np.concatenate(y_pred, axis=0)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == mnist.test.labels)

    print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
    np.save('pred.npy', y_pred)
    print('Output saved at pred.npy')

if __name__ == '__main__':
    
    model_dir = "models/a_very_robust_model"
    eval_batch_size = 64

    checkpoint = tf.train.latest_checkpoint(model_dir)
    x_adv = np.load("attack.npy")

    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 784) / 255.0

    if checkpoint is None:
        print('No checkpoint found')
    elif x_adv.shape != (10000, 784):
        print('Invalid shape: expected (10000,784), found {}'.format(x_adv.shape))
    elif np.amax(x_adv) > 1.0001 or np.amin(x_adv) < -0.0001 or np.isnan(np.amax(x_adv)):
        print('Invalid pixel range. Expected [0, 1], found [{}, {}]'.format(
            np.amin(x_adv), np.amax(x_adv)))
    else:
        model = Model()
        optimizer = tf.keras.optimizers.Adam()

        # Load the checkpoint for the clean model
        model.load_weights(checkpoint)

        # Initialize the LinfPGDAttack for adversarial examples
        attack = LinfPGDAttack(model, epsilon=0.3, k=40, a=0.01, random_start=True, loss_func=tf.nn.sparse_softmax_cross_entropy_with_logits)

        # Training with both clean and adversarial examples
        batch_size = 64
        num_epochs = 5
        num_batches = len(x_test) // batch_size

        for epoch in range(num_epochs):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size

                x_batch_clean = x_test[start_idx:end_idx]
                y_batch = y_test[start_idx:end_idx]

                # Create adversarial examples for this batch
                x_batch_adv = attack.perturb(x_batch_clean, y_batch)

                # Concatenate clean and adversarial batches
                x_batch = np.concatenate([x_batch_clean, x_batch_adv], axis=0)
                y_batch = np.concatenate([y_batch, y_batch], axis=0)

                # Shuffle the combined batch to avoid overfitting to adversarial examples
                idx = np.random.permutation(2 * batch_size)
                x_batch = x_batch[idx]
                y_batch = y_batch[idx]

                # Convert to TensorFlow tensors
                x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                y_batch = tf.convert_to_tensor(y_batch, dtype=tf.int32)

                # Training step with adversarial examples
                train_step_with_adversarial(x_batch, y_batch, model, optimizer, attack)

            # Evaluate accuracy on clean and adversarial test sets after each epoch
            x_adv_test = attack.perturb(x_test, y_test)
            clean_accuracy = model.calculate_accuracy(x_test, y_test)
            adv_accuracy = model.calculate_accuracy(x_adv_test, y_test)

            print(f"Epoch {epoch + 1}:")
            print(f"    Clean Test Accuracy: {clean_accuracy:.2f}%")
            print(f"    Adversarial Test Accuracy: {adv_accuracy:.2f}%")

        # Save the model after training
        model.save_weights(os.path.join(model_dir, 'adversarial_model'))