import numpy as np
from models.arachnet import ArachNet
from models.arachnet import UnetSubnet
from utils.arachnet_utils import segmentation_feature_extraction, GenImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import SGD
from utils.arachnet_utils import dice_coef_loss

arachnet = ArachNet(init_size=64, subnet_act="tanh", epu_act="sigmoid", features_num=2,
                                  subnet=UnetSubnet)

train_x = None # this should be a list or np.ndarray of images
train_y = None # this should be a list or np.ndarray of images

train_datagen = GenImageDataGenerator(x=train_x, y=train_y, batch_size=32,
                                      featurewise_center=False,
                                      samplewise_center=False,
                                      featurewise_std_normalization=False,
                                      samplewise_std_normalization=False,
                                      zca_whitening=False,
                                      zca_epsilon=1e-6,
                                      rotation_range=30,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      brightness_range=None,
                                      shear_range=0.,
                                      zoom_range=0.,
                                      channel_shift_range=0.,
                                      fill_mode='nearest',
                                      cval=0.,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      rescale=None,
                                      preprocessing_function=None,
                                      data_format=None,
                                      validation_split=0.0,
                                      dtype=np.float32,
                                      preprocess=segmentation_feature_extraction,
                                      multiple_outputs=arachnet.features_num)

valid_x = None # this should be a list or np.ndarray of images or list of fnames
valid_y = None # this should be a list or np.ndarray of images or list of fnames
valid_datagen = GenImageDataGenerator(x=valid_x, y=valid_y, batch_size=32,
                                      featurewise_center=False,
                                      samplewise_center=False,
                                      featurewise_std_normalization=False,
                                      samplewise_std_normalization=False,
                                      zca_whitening=False,
                                      zca_epsilon=1e-6,
                                      rotation_range=15,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      brightness_range=None,
                                      shear_range=0.,
                                      zoom_range=0.,
                                      channel_shift_range=0.,
                                      fill_mode='nearest',
                                      cval=0.,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      rescale=None,
                                      preprocessing_function=None,
                                      data_format=None,
                                      validation_split=0.0,
                                      dtype=np.float32,
                                      preprocess=segmentation_feature_extraction,
                                      multiple_outputs=arachnet.features_num)

es = EarlyStopping(monitor="val_loss", patience=50, verbose=1, restore_best_weights=True)

learning_rate = 0.01
momentum = 0.9
lr_decay = 1e-6
lr_drop = 20


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = LearningRateScheduler(lr_scheduler)
optimizer = SGD(learning_rate=learning_rate, decay=lr_decay,
                momentum=momentum, nesterov=True)

epu.compile(optimizer=optimizer, loss=[dice_coef_loss, "mse", "mse"], metrics=[config["metric"]],
            run_eagerly=False)
epu.fit(x=train_datagen, epochs=3000, validation_data=valid_datagen,
        batch_size=32,
        callbacks=[es])

np.save("../trained_models/arachnet.npy", arachnet.get_weights())