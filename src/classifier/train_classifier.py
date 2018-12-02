import tensorflow as tf
from src.classifier.NetworkData import NetworkData
from src.classifier.Classifier import Classifier

mnist = tf.keras.datasets.mnist
(x_train_data, y_train_data), (x_test_data, y_test_data) = mnist.load_data()

# Separación de las muestras
x_train = x_train_data[0:2000]
y_train = y_train_data[0:2000]

x_val_1 = x_train_data[2000:2200]
y_val_1 = y_train_data[2000:2200]

x_val_2 = x_train_data[2200:2400]
y_val_2 = y_train_data[2200:2400]

x_test = x_test_data[0:10000]
y_test = y_test_data[0:10000]

print('''Se cargaron {} muestras de entrenamiento, {} para el primer conjunto de validación, 
{} para el segundo conjunto de validación, y {} muestras de testeo.'''
      .format(len(x_train), len(x_val_1), len(x_val_2), len(x_test)))

print("Muestras de entrenamiento del 0: {}".format(y_train.tolist().count(0)))
print("Muestras de entrenamiento del 1: {}".format(y_train.tolist().count(1)))
print("Muestras de entrenamiento del 2: {}".format(y_train.tolist().count(2)))
print("Muestras de entrenamiento del 3: {}".format(y_train.tolist().count(3)))
print("Muestras de entrenamiento del 4: {}".format(y_train.tolist().count(4)))
print("Muestras de entrenamiento del 5: {}".format(y_train.tolist().count(5)))
print("Muestras de entrenamiento del 6: {}".format(y_train.tolist().count(6)))
print("Muestras de entrenamiento del 7: {}".format(y_train.tolist().count(7)))
print("Muestras de entrenamiento del 8: {}".format(y_train.tolist().count(8)))
print("Muestras de entrenamiento del 9: {}".format(y_train.tolist().count(9)))


network_data = NetworkData()
network_data.checkpoint_path = "out/checkpoint/model.ckpt"
network_data.model_path = "out/model/model.pb"
network_data.tensorboard_path = "out/tensorboard/"

network_data.num_features = 28
network_data.num_classes = 10

network_data.num_h1_units = 512
network_data.h1_activation = tf.nn.relu
# network_data.h1_kernel_init = tf.truncated_normal_initializer(stddev=0.1)
# network_data.h1_bias_init = tf.zeros_initializer()

network_data.num_h2_units = 256
network_data.h2_activation = tf.nn.relu
# network_data.h2_kernel_init = tf.truncated_normal_initializer(stddev=0.1)
# network_data.h2_bias_int = tf.zeros_initializer()

network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.0001

network_data.regularizer = 0.5

net = Classifier(network_data)

net.create_graph()
net.train(
    train_features=x_train,
    train_labels=y_train,
    val_features=x_val_1,
    val_labels=y_val_1,
    restore_run=False,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=5,
    training_epochs=100,
    batch_size=10
)

net.validate(x_val_2, y_val_2, batch_size=1)

# # for i in range(10):
# #     print("{} - {}".format(y_test[i], net.predict(x_test[i])))
