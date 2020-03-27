# encoding=utf-8
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, BatchNormalization, Layer, UpSampling2D, Multiply, GlobalMaxPooling2D
from keras.layers.core import Dropout, Lambda, Flatten, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.activations import relu
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


# AttentionGate
class AttentionGate(Layer):
    """docstring for ClassName"""

    def __init__(self, output_dim, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        super(AttentionGate, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        g, x = x
        x_ = Conv2D(self.output_dim, (1, 1))(x)
        x_ = MaxPooling2D((2, 2))(x_)
        g_ = Conv2D(self.output_dim, (1, 1))(g)
        g_ = relu(x_ + g_)
        g_ = Conv2D(1, (1, 1), activation='sigmoid')(g_)
        a = UpSampling2D((2, 2), interpolation='bilinear')(g_)
        return a

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_g, shape_x = input_shape
        return shape_x


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


# unet with attention gate
# 默认值参数，调用时赋值可更改
def aunet(IMG_WIDTH=224, IMG_HEIGHT=160, IMG_CHANNELS=1, pretrained_weights=False):
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

    a6 = AttentionGate(128, name='a6')([c5, c4])
    a6 = Multiply()([c4, a6])
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, a6])
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c6)

    a7 = AttentionGate(64, name='a7')([c6, c3])
    a7 = Multiply()([c3, a7])
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, a7])
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c7)

    a8 = AttentionGate(32, name='a8')([c7, c2])
    a8 = Multiply()([c2, a8])
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, a8])
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


# 函数名原 get_unet 改为 unet
# 默认值参数，调用时赋值可更改
def regression_net(IMG_WIDTH=224, IMG_HEIGHT=160, IMG_CHANNELS=1, pretrained_weights=False):
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
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
    g5 = GlobalMaxPooling2D()(p5)

    f6 = Dense(2048, activation='relu')(g5)
    f7 = Dense(2048, activation='relu')(f6)

    outputs = Dense(1, name='output')(f7)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer='adadelta', loss='mse', metrics=['mae', 'mse'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# 函数名原 get_unet 改为 unet
# 默认值参数，调用时赋值可更改
def multi_net(IMG_WIDTH=224, IMG_HEIGHT=160, IMG_CHANNELS=1, pretrained_weights=False):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)  # 将任意表达式封装为 Layer 对象。除以255.
    # keras.initializers.he_normal(seed=None)   Keras层的初始随机权重
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # (80, 112)

    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # (40, 56)

    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # (20, 28)

    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)  # (10, 14)

    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c5)

    a6 = AttentionGate(128, name='a6')([c5, c4])
    a6 = Multiply()([c4, a6])
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, a6])
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c6)

    a7 = AttentionGate(64, name='a7')([c6, c3])
    a7 = Multiply()([c3, a7])
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, a7])
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c7)

    a8 = AttentionGate(32, name='a8')([c7, c2])
    a8 = Multiply()([c2, a8])
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, a8])
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

    final_output = Conv2D(1, (1, 1), activation='sigmoid',
                          name='final_output')(c9)

    # regression network
    r5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p4)
    r5 = Dropout(0.3)(r5)
    r5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(r5)
    p5 = MaxPooling2D((2, 2))(r5)  # (5, 7)
    g5 = GlobalMaxPooling2D()(p5)

    f6 = Dense(2048, activation='relu')(g5)
    f7 = Dense(2048, activation='relu')(f6)

    regress_output = Dense(1, name='regress_output')(f7)

    model = Model(inputs=[inputs], outputs=[regress_output, final_output])
    # model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'regress_output': 'mse', 'final_output': focal_loss},
                  loss_weights={'regress_output': 1., 'final_output': 1.},
                  metrics={'regress_output': ['mae', 'mse'], 'final_output': dice_coef})

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

# 函数名原 get_unet 改为 unet
# 默认值参数，调用时赋值可更改
def multi_aunet(IMG_WIDTH=224, IMG_HEIGHT=160, IMG_CHANNELS=1, pretrained_weights=False):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)  # 将任意表达式封装为 Layer 对象。除以255.
    # keras.initializers.he_normal(seed=None)   Keras层的初始随机权重
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # (80, 112)

    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # (40, 56)

    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # (20, 28)

    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)  # (10, 14)

    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c5)

    a6 = AttentionGate(128, name='a6')([c5, c4])
    a6 = Multiply()([c4, a6])
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, a6])
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c6)

    a7 = AttentionGate(64, name='a7')([c6, c3])
    a7 = Multiply()([c3, a7])
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, a7])
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c7)

    a8 = AttentionGate(32, name='a8')([c7, c2])
    a8 = Multiply()([c2, a8])
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, a8])
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

    final_output = Conv2D(1, (1, 1), activation='sigmoid',
                          name='final_output')(c9)

    # regression network
    p5 = MaxPooling2D((2, 2))(c5)  # (5, 7)
    r6 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p5)
    r6 = Dropout(0.3)(r6)
    r6 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(r6)
    g6 = GlobalMaxPooling2D()(r6)

    f7 = Dense(2048, activation='relu')(g6)
    f8 = Dense(2048, activation='relu')(f7)

    regress_output = Dense(1, name='regress_output')(f8)

    model = Model(inputs=[inputs], outputs=[regress_output, final_output])
    # model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'regress_output': 'mse', 'final_output': focal_loss},
                  loss_weights={'regress_output': 1., 'final_output': 1.},
                  metrics={'regress_output': ['mae', 'mse'], 'final_output': dice_coef})

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


# 函数名原 get_unet 改为 unet
# 默认值参数，调用时赋值可更改
def multi_unet(IMG_WIDTH=224, IMG_HEIGHT=160, IMG_CHANNELS=1, pretrained_weights=False):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)  # 将任意表达式封装为 Layer 对象。除以255.
    # keras.initializers.he_normal(seed=None)   Keras层的初始随机权重
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # (80, 112)

    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  # (40, 56)

    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)  # (20, 28)

    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)  # (10, 14)

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

    final_output = Conv2D(1, (1, 1), activation='sigmoid',
                          name='final_output')(c9)

    # regression network
    p5 = MaxPooling2D((2, 2))(c5)  # (5, 7)
    r6 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p5)
    r6 = Dropout(0.3)(r6)
    r6 = Conv2D(256, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(r6)
    g6 = GlobalMaxPooling2D()(r6)

    f7 = Dense(2048, activation='relu')(g6)
    f8 = Dense(2048, activation='relu')(f7)

    regress_output = Dense(1, name='regress_output')(f8)

    model = Model(inputs=[inputs], outputs=[regress_output, final_output])
    # model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'regress_output': 'mse', 'final_output': focal_loss},
                  loss_weights={'regress_output': 1., 'final_output': 1.},
                  metrics={'regress_output': ['mae', 'mse'], 'final_output': dice_coef})

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def multi_unet_dm_origin(IMG_WIDTH=224, IMG_HEIGHT=160, IMG_CHANNELS=1, pretrained_weights=False):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    c1 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  # (80, 112)

    c2 = Conv2D(32, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c2)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(c2)

    c4 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(u3)

    u4 = concatenate([c1, c4])
    c5 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(u4)
    c5 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c5)

    final_output = Conv2D(1, (1, 1), activation='sigmoid',
                          name='final_output')(c5)

    # regression network
    p2 = MaxPooling2D((2, 2))(c2)  # (5, 7)
    r3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p2)
    r3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(r3)
    r3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(r3)
    r3 = Conv2D(64, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(r3)
    p3 = MaxPooling2D((2, 2))(r3)
    r4 = Conv2D(128, (3, 3), activation='elu',
                kernel_initializer='he_normal', padding='same')(p3)
    g4 = GlobalMaxPooling2D()(r4)

    f5 = Dense(256, activation='relu')(g4)
    f6 = Dense(128, activation='relu')(f5)

    regress_output = Dense(1, name='regress_output')(f6)

    model = Model(inputs=[inputs], outputs=[regress_output, final_output])
    # model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'regress_output': 'mse', 'final_output': 'mse'},
                  loss_weights={'regress_output': 1., 'final_output': 1},
                  metrics={'regress_output': ['mae', 'mse'], 'final_output': 'mse'})

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model