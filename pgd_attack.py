import tensorflow as tf

class LinfPGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_func = loss_func  # Accept the loss function as a callable

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + tf.random.uniform(shape=x_nat.shape, minval=-self.epsilon, maxval=self.epsilon)
            x = tf.clip_by_value(x, 0, 1)  # ensure valid pixel range
        else:
            x = tf.identity(x_nat)

        accuracy_adv = self.model.calculate_accuracy(x, y)

        for i in range(self.k):
            grad = self.compute_gradients(x, y)

            x += self.a * tf.sign(grad)

            x = tf.clip_by_value(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = tf.clip_by_value(x, 0, 1)  # ensure valid pixel range

        accuracy_perturbed = self.model.calculate_accuracy(x, y)

        return x

    def compute_gradients(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            pre_softmax = self.model(x, training=False)  # Use the model directly to get pre_softmax
            loss = self.loss_func(y, pre_softmax)

        grad = tape.gradient(loss, x)
        return grad
