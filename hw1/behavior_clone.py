
"""
Homework 1
Section 2 Warm up
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json, io


class BehaviorClone:
    def __init__(self, observation_space, acton_space, mode_directory, record_losses=True):
        self._observation_space = observation_space
        self._action_space = acton_space
        self.losses = []
        self._record_losses = record_losses
        self._mode_directory = mode_directory
        self._model = None

    def predict(self, observations):
        observations = np.array(observations)
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=observations,
            shuffle=False)
        return self.predict(input_fn)

    def predict(self, input_fn):
        if self._model is None:
            self._model = tf.estimator.Estimator(
                model_fn=lambda features, labels, mode: self.__nn_model_fn(features, labels, mode),
                model_dir=self._mode_directory)
        return self._model.predict(input_fn=input_fn)

    def learn(self, observations, actions):
        observations = np.array(observations, np.float64)
        actions = np.array(actions, np.float64)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=observations,
            y=actions,
            num_epochs=None,
            batch_size=len(actions),
            shuffle=False)
        nn_classifier = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode: self.__nn_model_fn(features, labels, mode),
            model_dir=self._mode_directory)

        class MyHook(tf.train.SessionRunHook):
            def __init__(self):
                self.losses = []

            def before_run(self, run_context):
                _session = run_context.session
                t = _session.graph.as_graph_element('loss_tensor:0')
                return tf.train.SessionRunArgs(t)

            def after_run(self, run_context, run_value):
                self.losses.append(run_value.results)

        hook = MyHook()
        tensors_to_log = {"step loss": "loss_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
        hooks = []
        if self._record_losses:
            hooks = [hook]
        nn_classifier.train(train_input_fn, steps=200, hooks=hooks)
        self.losses = hook.losses

    def show_train_losses(self):
        x = range(1, len(self.losses)+1)
        plt.plot(x, self.losses)
        plt.xlabel('training step')
        plt.ylabel('loss')
        plt.show()

    def __nn_model_fn(self, features, labels, mode):
        dens_layer1 = tf.layers.dense(inputs=features, units=128, activation=tf.nn.relu)
        dens_layer2 = tf.layers.dense(inputs=dens_layer1, units=64, activation=tf.nn.relu)
        logit_layer = tf.layers.dense(inputs=dens_layer2, units=self._action_space)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=logit_layer)

        loss = tf.losses.mean_squared_error(labels=labels, predictions=logit_layer)
        tf.identity(loss, name='loss_tensor')

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_ops = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss,train_op=train_ops)


if __name__ == '__main__':
    with io.open('experts/RoboschoolHopper-v1.json', 'r') as fp:
        expert_data = json.load(fp)
        behavior_clone = BehaviorClone(expert_data['observationSpace'], expert_data['actionSpace'],
                                       'experts/RoboschoolHopper-v1')
        behavior_clone.learn(expert_data['observations'], expert_data['actions'])
        behavior_clone.show_train_losses()