import numpy as np
import tensorflow as tf

from subnetworks import UnetSubnet
from tensorflow.keras.layers import Activation


class ArachNet(tf.keras.Model):

    def __init__(self, input_size=(96, 96, 1), init_size=32, subnet_act="sigmoid", arach_act="sigmoid", subnet=UnetSubnet,
                 features_num=5):

        super(ArachNet, self).__init__()
        self.sub_nets = []
        self.features_num = features_num
        self.height, self.width, self.depth = input_size

        # bias
        self.b = tf.Variable(tf.zeros([self.height, self.width, self.depth]), trainable=True)
        self.__interpret_output = []

        # Initialization of feature networks
        for batch in range(self.features_num):
            self.sub_nets.append(subnet(init_size=init_size, act=subnet_act))

        self.activate = Activation(arach_act)

    def call(self, x, **kwargs):
        _outputs = []

        for i, feature in enumerate(x):
            _outputs.append(self.sub_nets[i](feature))

        summation = tf.reduce_sum(_outputs, 0) + self.b

        self.__interpret_output = _outputs
        _output = self.activate(summation)
        return [_output, *_outputs]

    def get_interpret_output(self):
        return self.__interpret_output

    def segmentation_map(self, image, feature_name=""):
        pass

    def get_statistics(self, image):
        pass

    def get_dataset_statistics(self, dataset):
        pass

    def visualize_interpretations(self):
        interpretations = []

        for i, feature in enumerate(self.__interpret_output):
            feature = feature.numpy()[0]
            feature = feature.reshape(feature.shape[0], feature.shape[1])
            interpretation = np.zeros((feature.shape[0], feature.shape[1], 3), dtype=np.uint8)
            negative_contributions = ((feature < 0) * feature)
            negative_contributions = (negative_contributions * -1) * 255
            positive_contributions = ((feature > 0) * feature) * 255
            interpretation[:, :, 1] = positive_contributions
            interpretation[:, :, 2] = negative_contributions
            interpretations.append(interpretation)

        return interpretations
