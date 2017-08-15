import numpy as np
import tensorflow as tf

from isan_cell import ISAN


class ParenthesisTask(object):
    def __init__(self, max_count=5, implied_activation_fn="softmax"):
        """
        Saturating count number of non-closed parenthesis.

        Args:
          max_count: level at which outstanding opening parenthesis stop being added
          implied_activation_fn: how is the output of the netowkr interpreted:
            - softmax: treat it as logits, train via neg-log likelihood minimization
            - identity: treat it as probabilites, train via least-squares
        """
        self.max_count = max_count
        self.implied_activation_fn = implied_activation_fn
        self.parens = "()[]{}"
        self.n_paren_types = len(self.parens) // 2
        self.noises = "a"

        self.id_to_token = self.parens + self.noises
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        self.n_tokens = len(self.id_to_token)
        self.n_outputs = self.n_paren_types * (self.max_count + 1)

    def sample_batch(self, t, bs):
        inputs = (np.random.rand(bs, t) * len(self.id_to_token)).astype(np.int32)
        counts = np.zeros((bs, self.n_paren_types), dtype=np.int32)
        targets = np.zeros((bs, t, self.n_paren_types), dtype=np.int32)
        opening_parens = (np.arange(0, self.n_paren_types) * 2)[None, :]
        closing_parens = opening_parens + 1
        for i in range(t):
            opened = np.equal(inputs[:, i, None], opening_parens)
            counts = np.minimum(self.max_count, counts + opened)
            closed = np.equal(inputs[:, i, None], closing_parens)
            counts = np.maximum(0, counts - closed)
            targets[:, i, :] = counts
        return inputs, targets

    def loss(self, logits, targets):
        bs, t, _ = logits.get_shape().as_list()
        logits = tf.reshape(logits, (bs, t, self.n_paren_types, self.max_count + 1))

        if self.implied_activation_fn == "softmax":
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
        elif self.implied_activation_fn == "identity":
            loss = tf.square(logits - tf.one_hot(targets, self.max_count + 1))
        else:
            raise Exception()

        return tf.reduce_mean(loss)

    def print_batch(self, inputs, targets, predictions=None, max_to_print=1):
        if predictions is not None:
            predictions = np.reshape(
                predictions,
                predictions.shape[:2] + (self.n_paren_types, self.max_count + 1))
        for i in range(min(max_to_print, inputs.shape[1])):
            print("%3d:" % (i,), " ".join([self.id_to_token[t] for t in inputs[:, i]]))
            for paren_kind in range(self.n_paren_types):
                print("G%s:" % self.parens[2 * paren_kind:2 * paren_kind + 2],
                      " ".join([str(c) for c in targets[:, i, paren_kind]]))
                if predictions is not None:
                    pred = np.argmax(predictions[:, i, paren_kind], axis=1)
                    print("P%s:" % self.parens[2 * paren_kind:2 * paren_kind + 2],
                          " ".join([str(c) for c in pred]))


def main():
    MAX_COUNT = 5
    HIDDEN_DIM = 50  # will work nicely if set to 2*(MAX_COUNT + 1)
    N_TRAIN_STEPS = 20000
    BATCH_SIZE = 16
    T = 100
    WEIGHT_DECAY = 1e-4
    task = ParenthesisTask(max_count=MAX_COUNT,
                           # implied_activation_fn="softmax")
                           implied_activation_fn="identity")
    task.print_batch(*task.sample_batch(100, 2))

    tf.reset_default_graph()

    with tf.variable_scope("model"):
        model = ISAN(task.n_tokens, HIDDEN_DIM, task.n_outputs)

        inputs = tf.placeholder(tf.int32, shape=(BATCH_SIZE, T), name="inputs")
        targets = tf.placeholder(tf.int32, shape=(BATCH_SIZE, T, task.n_paren_types),
                                 name="targets")

        logits = model.fprop(inputs)

        task_loss = task.loss(logits, targets)
        weight_loss = 0.0
        for v in tf.trainable_variables():
            if v.name.split('/')[-1].startswith('W'):
                weight_loss = weight_loss + WEIGHT_DECAY * tf.nn.l2_loss(v)

        loss = task_loss + weight_loss

    learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    tf.get_variable_scope().reuse_variables()

    print(inputs.get_shape().as_list())

    print("Training the following variables:")
    for v in tf.trainable_variables():
        print("%s: %s (%s)" % (v.name, v.get_shape().as_list(),
                               v.initializer.inputs[1].op.name))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        PRINT_PERIOD = 500

        task_loss_acc = 0

        for i in range(N_TRAIN_STEPS):
            v_inputs, v_targets = task.sample_batch(T, BATCH_SIZE)
            feed_dict = {inputs: v_inputs,
                         targets: v_targets,
                         # learning_rate:min(1e-4, 1e-4 * 1000/(i+1))
                         learning_rate: 2e-4
                         }
            v_task_loss, v_logits, _ = sess.run([task_loss, logits, train_op], feed_dict=feed_dict)

            task_loss_acc += v_task_loss

            if ((i + 1) % PRINT_PERIOD) == 0:
                print("Step %d loss %f" % (i, task_loss_acc / PRINT_PERIOD))
                task.print_batch(v_inputs, v_targets, v_logits)
                print("")
                task_loss_acc = 0


if __name__ == '__main__':
    main()
