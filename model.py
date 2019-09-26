import tensorflow as tf


def get_model(input_shape, output_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=4, kernel_size=3,
                               activation=tf.nn.relu, padding='same'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3,
                               activation=tf.nn.relu),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3,
                               activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax)
    ])

    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(output_shape, activation=tf.nn.softmax)
    ])
    '''

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_model(model_path):
    return tf.keras.models.load_model(model_path)
