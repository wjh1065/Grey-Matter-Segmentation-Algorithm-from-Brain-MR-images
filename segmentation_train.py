import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Conv3D, MaxPooling3D, concatenate, \
    Conv3DTranspose, ReLU, Cropping3D, LeakyReLU, PReLU, UpSampling3D, Activation, add
from tensorflow.keras import activations
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Nadam, Adadelta
from tensorflow.keras.models import load_model
from tqdm import tqdm_notebook, tqdm
import nibabel as nib
import numpy as np
import os
import shutil
import pandas as pd

"""
multi gpu
"""
m = 'seg_ADNI_002_Vnet_epoch1000_smooth_30_filter10.h5'  # model name
l = 'seg_ADNI_002_Vnet_epoch1000_smooth_30_filter10.png'  # Loss_graph name
c = 'seg_ADNI_002_Vnet_epoch1000_smooth_30_filter10.csv'  # Loss_csv name

"""
make patch
"""


def get_patches(img_arr, size=128, stride=128):
    patched_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1
    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping
        for i in range(i_max):
            for j in range(i_max):
                for k in range(i_max):
                    patched_list.append(img_arr[i * stride: i * stride + size, j * stride: j * stride + size,
                                        k * stride: k * stride + size, ])
    else:
        raise ValueError("img_arr.ndim must be equal 4")
    return np.stack(patched_list)


"""
load input, target // nii.gz
"""


def data_list_input(root):
    dir_list = os.listdir(root)
    list_input = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_input.append(data_path + '/Input_T1.nii.gz')
    return list_input


def data_list_output(root):
    dir_list = os.listdir(root)
    list_output = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_output.append(data_path + '/Target_gm_mask.nii.gz')
    return list_output


def data_load_input(file_list):
    patches = []
    for i in tqdm(file_list):
        load_input = nib.load(i)
        load_input = load_input.get_fdata()
        load_input = np.array(load_input, dtype=np.float32)
        max_val = load_input.max()
        print('input_data max : ', max_val)
        print('subject : ', i)
        min_val = load_input.min()
        print('input_data min : ', min_val)
        normalized_load_input_1 = load_input / max_val
        normalized_load_input = get_patches(img_arr=normalized_load_input_1, size=128, stride=128)
        patches.append(normalized_load_input)
    patches = np.vstack(patches)
    print('input patches shape : ', patches.shape)
    return patches


def data_load_output(file_list):
    patches = []
    for i in tqdm(file_list):
        load_output = nib.load(i)
        load_output = load_output.get_fdata()
        load_output = np.array(load_output, dtype=np.float32)
        max_val = load_output.max()
        print('input_data max : ', max_val)
        print('subject : ', i)
        min_val = load_output.min()
        print('input_data min : ', min_val)
        normalized_load_output_1 = load_output / max_val
        normalized_load_output = get_patches(img_arr=normalized_load_output_1, size=128, stride=128)
        patches.append(normalized_load_output)
    patches = np.vstack(patches)
    print('output patches shape : ', patches.shape)
    return patches


"""
load "val" input, target // nii.gz
"""


def data_list_val_input(root):
    dir_list = os.listdir(root)
    list_input = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_input.append(data_path + '/Input_T1.nii.gz')
    return list_input


def data_list_val_output(root):
    dir_list = os.listdir(root)
    list_output = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            list_output.append(data_path + '/Target_gm_mask.nii.gz')
    return list_output


def data_load_val_input(file_list):
    patches = []
    for i in tqdm(file_list):
        load_input = nib.load(i)
        load_input = load_input.get_fdata()
        load_input = np.array(load_input, dtype=np.float32)
        max_val = load_input.max()
        print('input_data max : ', max_val)
        print('subject : ', i)
        min_val = load_input.min()
        print('input_data min : ', min_val)
        normalized_load_input_1 = load_input / max_val
        normalized_load_input = get_patches(img_arr=normalized_load_input_1, size=128, stride=128)
        patches.append(normalized_load_input)
    patches = np.vstack(patches)
    print('val_input patches shape : ', patches.shape)
    return patches


def data_load_val_output(file_list):
    patches = []
    for i in tqdm(file_list):
        load_output = nib.load(i)
        load_output = load_output.get_fdata()
        load_output = np.array(load_output, dtype=np.float32)
        max_val = load_output.max()
        print('input_data max : ', max_val)
        print('subject : ', i)
        min_val = load_output.min()
        print('input_data min : ', min_val)
        normalized_load_output_1 = load_output / max_val
        normalized_load_output = get_patches(img_arr=normalized_load_output_1, size=128, stride=128)
        patches.append(normalized_load_output)
    patches = np.vstack(patches)
    print('val_output patches shape : ', patches.shape)
    return patches


def lcs_4floor_Resunet3D(filters=10):
    inputs = Input((128, 128, 128, 1))

    conv = Conv3D(filters * 2, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(conv)
    shortcut = Conv3D(filters * 4, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(inputs)
    shortcut = BatchNormalization()(shortcut)
    output1 = add([shortcut, conv])

    res1 = BatchNormalization()(output1)
    res1 = Activation("relu")(res1)
    res1 = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(res1)
    res1 = BatchNormalization()(res1)
    res1 = Activation("relu")(res1)
    res1 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(res1)
    shortcut1 = Conv3D(filters * 8, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output1)
    shortcut1 = BatchNormalization()(shortcut1)
    output2 = add([shortcut1, res1])

    res2 = BatchNormalization()(output2)
    res2 = Activation("relu")(res2)
    res2 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(res2)
    res2 = BatchNormalization()(res2)
    res2 = Activation("relu")(res2)
    res2 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(res2)
    shortcut2 = Conv3D(filters * 16, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output2)
    shortcut2 = BatchNormalization()(shortcut2)
    output3 = add([shortcut2, res2])

    res3 = BatchNormalization()(output3)
    res3 = Activation("relu")(res3)
    res3 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(res3)
    res3 = BatchNormalization()(res3)
    res3 = Activation("relu")(res3)
    res3 = Conv3D(filters * 32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(res3)
    shortcut3 = Conv3D(filters * 32, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output3)
    shortcut3 = BatchNormalization()(shortcut3)
    output4 = add([shortcut3, res3])

    # bridge
    conv = BatchNormalization()(output4)
    conv = Activation("relu")(conv)
    conv = Conv3D(filters * 32, kernel_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv3D(filters * 64, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(conv)
    shortcut5 = Conv3D(filters * 64, kernel_size=(1, 1, 1), padding='same', strides=(2, 2, 2))(output4)
    shortcut5 = BatchNormalization()(shortcut5)
    output_bd = add([shortcut5, conv])

    # decoder

    uconv2 = UpSampling3D((2, 2, 2))(output_bd)
    uconv2 = concatenate([uconv2, output4])

    uconv22 = BatchNormalization()(uconv2)
    uconv22 = Activation("relu")(uconv22)
    uconv22 = Conv3D(filters * 32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv22)
    uconv22 = BatchNormalization()(uconv22)
    uconv22 = Activation("relu")(uconv22)
    uconv22 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv22)
    shortcut6 = Conv3D(filters * 16, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv2)
    shortcut6 = BatchNormalization()(shortcut6)
    output7 = add([uconv22, shortcut6])

    uconv3 = UpSampling3D((2, 2, 2))(output7)
    uconv3 = concatenate([uconv3, output3])

    uconv33 = BatchNormalization()(uconv3)
    uconv33 = Activation("relu")(uconv33)
    uconv33 = Conv3D(filters * 16, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv33)
    uconv33 = BatchNormalization()(uconv33)
    uconv33 = Activation("relu")(uconv33)
    uconv33 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv33)
    shortcut7 = Conv3D(filters * 8, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv3)
    shortcut7 = BatchNormalization()(shortcut7)
    output8 = add([uconv33, shortcut7])

    uconv4 = UpSampling3D((2, 2, 2))(output8)
    uconv4 = concatenate([uconv4, output2])

    uconv44 = BatchNormalization()(uconv4)
    uconv44 = Activation("relu")(uconv44)
    uconv44 = Conv3D(filters * 8, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv44)
    uconv44 = BatchNormalization()(uconv44)
    uconv44 = Activation("relu")(uconv44)
    uconv44 = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv44)
    shortcut8 = Conv3D(filters * 4, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv4)
    shortcut8 = BatchNormalization()(shortcut8)
    output9 = add([uconv44, shortcut8])

    uconv5 = UpSampling3D((2, 2, 2))(output9)
    uconv5 = concatenate([uconv5, output1])

    uconv55 = BatchNormalization()(uconv5)
    uconv55 = Activation("relu")(uconv55)
    uconv55 = Conv3D(filters * 4, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv55)
    uconv55 = BatchNormalization()(uconv55)
    uconv55 = Activation("relu")(uconv55)
    uconv55 = Conv3D(filters * 2, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1))(uconv55)
    shortcut9 = Conv3D(filters * 2, kernel_size=(1, 1, 1), padding='same', strides=(1, 1, 1))(uconv5)
    shortcut9 = BatchNormalization()(shortcut9)
    output10 = add([uconv55, shortcut9])

    output_layer = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(output10)
    model = Model(inputs, output_layer)

    return model


"""
load train/val data
"""
root = './data/seg_ADNI_002/train'
val_root = './data/seg_ADNI_002/valid'

list_input = data_list_input(root)
list_val_input = data_list_val_input(val_root)

list_output = data_list_output(root)
list_val_output = data_list_val_output(val_root)

input_data = np.array(data_load_input(list_input))
output_data = np.array(data_load_output(list_output))

valid_input_data = np.array(data_load_val_input(list_val_input))
valid_output_data = np.array(data_load_val_output(list_val_output))

"""
model [input / output] / [val_input / val_output]
"""
input_data = np.expand_dims(input_data, axis=4)
output_data = np.expand_dims(output_data, axis=4)
print('model input_data : ', input_data.shape)
print('model output_data : ', output_data.shape)

valid_input_data = np.expand_dims(valid_input_data, axis=4)
valid_output_data = np.expand_dims(valid_output_data, axis=4)
print('model val_input_data : ', valid_input_data.shape)
print('model val_output_data : ', valid_output_data.shape)

from tensorflow.keras import backend as K
smooth=30

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)

# with strategy.scope():
model = lcs_4floor_Resunet3D()

model.summary()

opt = Adam(lr=1e-5)
model.compile(optimizer= opt, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])

history = model.fit(input_data, output_data, epochs=1000, batch_size=1, shuffle=True, verbose=1,
                    validation_data=(valid_input_data, valid_output_data))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
import matplotlib.pyplot as plt

plt.plot(x_len[30:], y_vloss[30:], marker='None', c='red', label="Validation-set Loss")
plt.plot(x_len[30:], y_loss[30:], marker='None', c='blue', label="Train-set Loss")
fig1 = plt.gcf()
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

hist_df = pd.DataFrame(history.history)

model.save('model.h5')  ################### model save ####################
fig1.savefig('loss_graph.png', dpi=300)  ################### loss graph save ####################
hist_csv_file = 'loss_csv.csv'  ################## history save #####################

# image = 'loss_graph.png'
# lcs.send_photo(chat_id = '1599496651', photo=open(image, 'rb'))


with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

shutil.move('model.h5', m)
shutil.move('loss_graph.png', l)
shutil.move('loss_csv.csv', c)

shutil.move(m, 'results')
shutil.move(l, 'results')
shutil.move(c, 'results')