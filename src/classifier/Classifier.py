import os
import random
import time
from tensorflow.python.framework import graph_io
import tensorflow as tf
from src.classifier import NetworkData
import numpy as np


class Classifier:
    def __init__(self, network_data: NetworkData):
        self.graph: tf.Graph = tf.Graph()
        self.input_feature = None
        self.input_label = None
        self.input_label_one_hot = None
        self.flat_input = None
        self.hidden_1 = None
        self.hidden_2 = None
        self.output = None
        self.output_no_activation = None
        self.output_class = None
        self.loss = None
        self.accuracy = None
        self.training_op = None

        self.checkpoint_saver = None
        self.merged_summary = None

        self.network_data: NetworkData = network_data

    def create_graph(self):

        with self.graph.as_default():
            with tf.name_scope("input_feature"):
                self.input_feature = tf.placeholder(tf.float32, shape=[None, 28, 28], name="input_feature")

            with tf.name_scope("input_label"):
                self.input_label = tf.placeholder(tf.int32, shape=[None], name="input_label")
                self.input_label_one_hot = tf.one_hot(self.input_label, depth=10, name="input_label_one_hot")

            with tf.name_scope("flat_input"):
                self.flat_input = tf.layers.flatten(self.input_feature)

            with tf.name_scope("hidden_layer_1"):
                self.hidden_1 = tf.layers.dense(self.flat_input,
                                                units=self.network_data.num_h1_units,
                                                activation=self.network_data.h1_activation,
                                           # kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           # bias_initializer=tf.zeros_initializer(),
                                           name="hidden_1")
                tf.summary.histogram('hidden_layer_1', self.hidden_1)

            with tf.name_scope("hidden_layer_2"):
                self.hidden_2 = tf.layers.dense(self.hidden_1,
                                                units=self.network_data.num_h2_units,
                                                activation=self.network_data.h2_activation,
                                           # kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                           # bias_initializer=tf.zeros_initializer(),
                                           name="hidden_2")
                tf.summary.histogram('hidden_layer_2', self.hidden_2)

            with tf.name_scope("output_layer"):
                self.output_no_activation = tf.layers.dense(self.hidden_2, units=10, activation=None,
                                         # kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                         # bias_initializer=tf.zeros_initializer(),
                                         name="output")
                self.output = tf.nn.softmax(self.output_no_activation)
                self.output_class = tf.argmax(self.output, 1, output_type=tf.int32)

            with tf.name_scope("loss"):
                logits_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_no_activation,
                                                                                        labels=self.input_label_one_hot))
                dense_loss = 0
                for var in tf.trainable_variables():
                    if 'kernel' in var.name:
                        dense_loss += tf.nn.l2_loss(var)

                self.loss = logits_loss + self.network_data.regularizer * dense_loss
                tf.summary.scalar('loss', self.loss)

            with tf.name_scope("accuracy"):
                self.accuracy = tf.divide(tf.reduce_sum(tf.cast(
                    tf.equal(self.output_class, self.input_label), tf.float32)),
                    tf.cast(tf.shape(self.input_label)[0], tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

            with tf.name_scope("training"):
                self.training_op = tf.train.AdamOptimizer(
                    learning_rate=self.network_data.learning_rate,
                    epsilon=self.network_data.adam_epsilon).minimize(self.loss)

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)
            self.merged_summary = tf.summary.merge_all()

    def save_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None:
            self.checkpoint_saver.save(sess, self.network_data.checkpoint_path)

    def load_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None and tf.gfile.Exists("{}.meta".format(self.network_data.checkpoint_path)):
            self.checkpoint_saver.restore(sess, self.network_data.checkpoint_path)
        else:
            session = tf.Session()
            session.run(tf.initialize_all_variables())

    def save_model(self, sess: tf.Session):
        if self.network_data.model_path is not None:
            drive, path_and_file = os.path.splitdrive(self.network_data.model_path)
            path, file = os.path.split(path_and_file)
            graph_io.write_graph(sess.graph, path, file, as_text=False)

    def create_batch(self, input_list, batch_size):
        num_batches = int(np.ceil(len(input_list) / batch_size))
        batch_list = []
        for _ in range(num_batches):
            if (_ + 1) * batch_size < len(input_list):
                aux = input_list[_ * batch_size:(_ + 1) * batch_size]
            else:
                aux = input_list[len(input_list)-batch_size:len(input_list)]

            batch_list.append(aux)

        return batch_list

    def train(self,
              train_features,
              train_labels,
              val_features,
              val_labels,
              batch_size: int,
              training_epochs: int,
              restore_run: bool = True,
              save_partial: bool = True,
              save_freq: int = 10,
              shuffle: bool=True,
              use_tensorboard: bool = False,
              tensorboard_freq: int = 50,
              val_batch_size: int = 10,
              val_freq: int = 10):

        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())

            if restore_run:
                self.load_checkpoint(sess)

            train_writer = None
            val_writer = None
            if use_tensorboard:
                if self.network_data.tensorboard_path is not None:
                    # Set up tensorboard summaries and saver
                    if tf.gfile.Exists(self.network_data.tensorboard_path + '/train') is not True:
                        tf.gfile.MkDir(self.network_data.tensorboard_path + '/train')
                    else:
                        tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/train')

                    # Set up tensorboard summaries and saver
                    if tf.gfile.Exists(self.network_data.tensorboard_path + '/val') is not True:
                        tf.gfile.MkDir(self.network_data.tensorboard_path + '/val')
                    else:
                        tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/val')

                train_writer = tf.summary.FileWriter("{}train".format(self.network_data.tensorboard_path), self.graph)
                train_writer.add_graph(sess.graph)
                val_writer = tf.summary.FileWriter("{}val".format(self.network_data.tensorboard_path), self.graph)
                val_writer.add_graph(sess.graph)

            loss_ep = 0
            acc_ep = 0
            val_loss_ep = 0
            val_acc_ep = 0
            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep = 0
                acc_ep = 0
                n_step = 0

                database = list(zip(train_features, train_labels))

                for batch in self.create_batch(database, batch_size):
                    batch_features, batch_labels = zip(*batch)

                    feed_dict = {
                        self.input_feature: batch_features,
                        self.input_label: batch_labels
                    }

                    loss, _, acc = sess.run([self.loss, self.training_op, self.accuracy], feed_dict=feed_dict)

                    loss_ep += loss
                    acc_ep += acc
                    n_step += 1
                loss_ep = loss_ep / n_step
                acc_ep = acc_ep / n_step

                if use_tensorboard:
                    if epoch % tensorboard_freq == 0 and self.network_data.tensorboard_path is not None:
                        train_database = list(zip(train_features, train_labels))
                        random.shuffle(train_database)
                        aux_train_features, aux_train_labels = zip(*train_database)

                        train_feed_dict = {
                            self.input_feature: aux_train_features[0:batch_size],
                            self.input_label: aux_train_labels[0:batch_size]

                        }
                        s = sess.run(self.merged_summary, feed_dict=train_feed_dict)
                        train_writer.add_summary(s, epoch)

                        val_database = list(zip(val_features, val_labels))
                        random.shuffle(val_database)
                        val_features, val_labels = zip(*val_database)

                        val_feed_dict = {
                            self.input_feature: val_features[0:val_batch_size],
                            self.input_label: val_labels[0:val_batch_size]

                        }
                        s = sess.run(self.merged_summary, feed_dict=val_feed_dict)
                        val_writer.add_summary(s, epoch)

                if save_partial:
                    if epoch % save_freq == 0:
                        self.save_checkpoint(sess)
                        self.save_model(sess)

                if shuffle:
                    aux_list = list(zip(train_features, train_labels))
                    random.shuffle(aux_list)
                    train_features, train_labels = zip(*aux_list)

                if val_freq is not None and epoch % val_freq == 0:
                    val_loss_ep = 0
                    val_acc_ep = 0
                    val_n_step = 0

                    val_database = list(zip(val_features, val_labels))

                    for val_batch in self.create_batch(val_database, val_batch_size):
                        val_batch_features, val_batch_labels = zip(*val_batch)

                        val_feed_dict = {
                            self.input_feature: val_batch_features,
                            self.input_label: val_batch_labels
                        }

                        val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict=val_feed_dict)

                        val_loss_ep += val_loss
                        val_acc_ep += val_acc
                        val_n_step += 1
                    val_loss_ep = val_loss_ep / val_n_step
                    val_acc_ep = val_acc_ep / val_n_step

                print("Epoch %d of %d, loss %f, acc %f, val_loss %f, val_acc %f, epoch time %.2fmin, ramaining time %.2fmin" %
                      (epoch + 1,
                       training_epochs,
                       loss_ep,
                       acc_ep,
                       val_loss_ep,
                       val_acc_ep,
                       (time.time()-epoch_time)/60,
                       (training_epochs-epoch-1)*(time.time()-epoch_time)/60))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

            return acc_ep, loss_ep

    def validate(self, features, labels, batch_size: int = 1):
        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            acum_acc = 0
            acum_loss = 0
            n_step = 0
            database = list(zip(features, labels))
            batch_list = self.create_batch(database, batch_size)
            for batch in batch_list:
                feature, label = zip(*batch)
                feed_dict = {
                    self.input_feature: feature,
                    self.input_label: label,
                }
                acc, loss = sess.run([self.accuracy, self.loss], feed_dict=feed_dict)

                acum_acc += acc
                acum_loss += loss
                n_step += 1
            print("Validation acc: %f, loss: %f" % (acum_acc/n_step, acum_loss/n_step))

            sess.close()

            return acum_acc/len(labels), acum_loss/len(labels)

    def predict(self, feature):

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            feed_dict = {
                self.input_feature: [feature],
            }

            predicted = sess.run(self.output_class, feed_dict=feed_dict)

            sess.close()

            return predicted[0]
