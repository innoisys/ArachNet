import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import Concatenate, LayerNormalization

class UnetSubnet(tf.keras.Model):

    def __init__(self, init_size, act="softmax", classes=10, fc_hidden_units=512):
        super(UnetSubnet, self).__init__()
        self.act = act
        self.conv_one_a = Conv2D(32, (3, 3), padding="same", activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv_one_b = Conv2D(64, (3, 3), padding="same", activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.conv_two_a = Conv2D(64, (3, 3), padding="same", activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv_two_b = Conv2D(128, (3, 3), padding="same", activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.conv_three_a = Conv2D(128, (3, 3), padding="same", activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv_three_b = Conv2D(256, (3, 3), padding="same", activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.conv_four_a = Conv2D(256, (3, 3), padding="same", activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv_four_b = Conv2D(512, (3, 3), padding="same", activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        # Decoder
        self.dec_conv_three_b = Conv2D(256, (3, 3), padding="same", activation="relu",
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dec_conv_three_a = Conv2D(256, (3, 3), padding="same", activation="relu",
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.dec_conv_two_b = Conv2D(128, (3, 3), padding="same", activation="relu",
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dec_conv_two_a = Conv2D(128, (3, 3), padding="same", activation="relu",
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.dec_conv_one_b = Conv2D(64, (3, 3), padding="same", activation="relu",
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dec_conv_one_a = Conv2D(64, (3, 3), padding="same", activation="relu",
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.dec_output_conv = Conv2D(1, (1, 1), padding="same", activation=act,
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        # self.batch_nroms = [LayerNormalization() for _ in range(3)]

    def call(self, x, **kwargs):  # implement forward pass
        x = self.conv_one_a(x)
        x_enc_first = self.conv_one_b(x)
        # x_enc_first = self.batch_nroms[0](x_enc_first)
        x = MaxPool2D(pool_size=(2, 2))(x_enc_first)

        x = self.conv_two_a(x)
        x_enc_second = self.conv_two_b(x)
        # x_enc_second = self.batch_nroms[1](x_enc_second)
        x = MaxPool2D(pool_size=(2, 2))(x_enc_second)

        x = self.conv_three_a(x)
        x_enc_third = self.conv_three_b(x)
        # x_enc_third = self.batch_nroms[2](x_enc_third)
        x = MaxPool2D(pool_size=(2, 2))(x_enc_third)

        x = self.conv_four_a(x)
        x = self.conv_four_b(x)
        # x = self.batch_nroms[3](x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Concatenate()([x, x_enc_third])
        x = self.dec_conv_three_b(x)
        x = self.dec_conv_three_a(x)
        # x = self.batch_nroms[4](x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Concatenate()([x, x_enc_second])
        x = self.dec_conv_two_b(x)
        x = self.dec_conv_two_a(x)
        # x = self.batch_nroms[5](x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Concatenate()([x, x_enc_first])
        x = self.dec_conv_one_b(x)
        x = self.dec_conv_one_a(x)
        # x = self.batch_nroms[6](x)
        output = self.dec_output_conv(x)

        return output
