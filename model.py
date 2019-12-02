# encoding=utf-8
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dropout, Lambda, Flatten, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras import backend as K


# smooth = 1.
smooth = 0.001

K.set_image_data_format('channels_last')  # 已设置，可省略


def get_model(input_shape, output_shape, model_type='FC'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    if model_type == 'FC':
        model = Sequential([
            BatchNormalization(),
            Flatten(input_shape=input_shape),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(output_shape, activation='softmax')
        ])
    elif model_type == 'CNN':
        from config import pad
        if pad == 5:
            model = Sequential([
                BatchNormalization(),
                Conv2D(filters=4, kernel_size=3, activation='relu', padding='same'),
                MaxPooling2D(),
                Conv2D(filters=8, kernel_size=3, activation='relu'),
                Conv2D(filters=16, kernel_size=3, activation='relu'),
                Dense(128, activation='relu'),
                Dense(output_shape, activation='softmax')
            ])
        elif pad == 10:
            model = Sequential([
                BatchNormalization(),
                Conv2D(filters=4, kernel_size=3, activation='relu', padding='same'),
                MaxPooling2D(),
                Conv2D(filters=8, kernel_size=3, activation='relu', padding='same'),
                MaxPooling2D(),
                Conv2D(filters=16, kernel_size=3, activation='relu'),
                Conv2D(filters=32, kernel_size=3, activation='relu'),
                Dense(128, activation='relu'),
                Dense(output_shape, activation='softmax')
            ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# Metric function
def dice_coef(y_true, y_pred):
    y_true /= 255.
    # 08/23补充！
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
# 可用 1-dice ……


# Tversky Metric function
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return 1.0 - K.pow((1 - pt_1), gamma)


# return funtion 会报错，取消嵌套
def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


# 函数名原 get_unet 改为 unet
# 默认值参数，调用时赋值可更改
def unet(IMG_WIDTH=224, IMG_HEIGHT=160, IMG_CHANNELS=1, pretrained_weights=False):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)  # 将任意表达式封装为 Layer 对象。除以255.
    # keras.initializers.he_normal(seed=None)   Keras层的初始随机权重
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    # Got inputs shapes: [(None, 224, 192, 16), (None, 225, 192, 16)]。估计225奇数不方便处理
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss=[focal_loss], metrics=[dice_coef])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
