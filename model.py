import tensorflow as tf
from tensorflow.keras.backend import set_session


def get_model(input_shape, output_shape, model_type='FC'):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))
    if model_type == 'FC':
        model = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax)
        ])
    elif model_type == 'CNN':
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Conv2D(filters=4, kernel_size=3,
        #                            activation=tf.nn.relu, padding='same'),
        #     tf.keras.layers.MaxPool2D(),
        #     tf.keras.layers.Conv2D(filters=8, kernel_size=3,
        #                            activation=tf.nn.relu),
        #     tf.keras.layers.Conv2D(filters=16, kernel_size=3,
        #                            activation=tf.nn.relu),
        #     tf.keras.layers.Dense(128, activation=tf.nn.relu),
        #     tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax)])
        model = tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(filters=4, kernel_size=3,
                                   activation=tf.nn.relu, padding='same'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(filters=8, kernel_size=3,
                                   activation=tf.nn.relu, padding='same'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3,
                                   activation=tf.nn.relu),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3,
                                   activation=tf.nn.relu),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_model(model_path):
    return tf.keras.models.load_model(model_path)
