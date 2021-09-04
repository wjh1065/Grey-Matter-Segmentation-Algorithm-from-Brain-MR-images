import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tqdm import tqdm
import nibabel as nib
import numpy as np
import time
import os
import math
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from tensorflow.keras.optimizers import Adam


os.environ["CUDA_VISIBLE_DEVICES"]='0'

root = './data/seg_ADNI_002/pred'

"""
custom loss function
"""
from tensorflow.keras import backend as K
smooth=80
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
                    patched_list.append(img_arr[i * stride: i * stride + size, j * stride: j * stride + size, k * stride: k * stride + size, ])
    else:
        raise ValueError("img_arr.ndim must be equal 4")
    return np.stack(patched_list)

"""
reconstruct patch data
"""
def reconstruct_patch(img_arr, org_img_size, stride=128, size=128):
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")
    if size is None:
        size = img_arr.shape[2]
    if stride is None:
        stride = size
    nm_layers = img_arr.shape[4]
    i_max = (org_img_size[0] // stride ) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride ) + 1 - (size // stride)
    k_max = (org_img_size[2] // stride ) + 1 - (size // stride)
    total_nm_images = img_arr.shape[0] // (i_max ** 3)
    images_list = []
    kk=0
    for img_count in range(total_nm_images):
        img_bg = np.zeros((org_img_size[0],org_img_size[1],org_img_size[2],nm_layers), dtype=img_arr[0].dtype)
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for layer in range(nm_layers):
                        img_bg[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        k * stride: k * stride + size,
                        layer,
                        ] = img_arr[kk, :, :, :, layer]
                    kk += 1
        images_list.append(img_bg)
    return np.stack(images_list)




"""
model evaluate
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

list_input = data_list_input(root)
list_output = data_list_output(root)
input_data = np.array(data_load_input(list_input))
output_data = np.array(data_load_output(list_output))
input_data = np.expand_dims(input_data, axis=4)
output_data = np.expand_dims(output_data, axis=4)
print('model evaluate input_data : ', input_data.shape)
print('model evaluate output_data : ', output_data.shape)


model = load_model('./results/seg_ADNI_002_Vnet_epoch500_smooth_80_filter8.h5',custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
opt = Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef])

score = model.evaluate(x=input_data,y=output_data, batch_size=1)

print('score : ', score)
print("Test lost: ", score[0])
print("Test IOU: ", score[1])
print("Test Dice Coefficient: ", score[2])

"""
load data //  pred LR to HR
"""
def pred_LR2HR(root):
    dir_list = os.listdir(root)
    merge = []
    for i in dir_list:
        if i.startswith('ADNI'):
            data_path = os.path.join(root, i)
            pred_list_LR= (data_path + '/Input_T1.nii.gz')
            merge.append(pred_list_LR)
    return merge

"""
load patch pred LR data 
"""
def data_load_pred_LR(root, file_list):
    for i in tqdm(file_list):
        print('i : ', i)
        load_input = nib.load(i[0:78])
        print('subject : ',i[0:78])
        load_input = load_input.get_fdata()
        load_input = np.array(load_input, dtype=np.float32)
        max_val = load_input.max()
        # #print('input_data max : ', max_val)
        min_val = load_input.min()
        normalized_load_input_1 = load_input / max_val
        normalized_load_input = get_patches(img_arr=normalized_load_input_1, size=128,stride=128)
        # #print('patches shape : ', normalized_load_input.shape)
        pred_data = np.expand_dims(normalized_load_input,axis=4)
        # #print('pred data shape : ', pred_data.shape)
        #
        model = load_model('./results/seg_ADNI_002_Vnet_epoch500_smooth_80_filter8.h5', custom_objects={'dice_coef_loss':dice_coef_loss, 'iou':iou, 'dice_coef':dice_coef})
        #
        pred = model.predict(pred_data, batch_size=1)



        #print('model predict done')
        x_reconstructed = reconstruct_patch(img_arr=pred, org_img_size=(256,256,256), stride=128)
        final_pred = np.squeeze(x_reconstructed)
        #print('reconstructed shape : ', x_reconstructed.shape)
        header = nib.load(i[0:78])
        header_1 = header.get_fdata()
        max_val = header_1.max()
        final_pred_1 = final_pred * max_val

        data_path = os.path.join(i[0:63])

        img = nib.Nifti1Image(final_pred_1, header.affine)
        img.to_filename(os.path.join(data_path, 'seg_ADNI_002_Vnet_epoch500_smooth_80_filter8.nii.gz'))
        print('pred save done')
    return img




# pred_list_LR = pred_LR2HR(root)

# pred_LR = np.array(data_load_pred_LR(root,pred_list_LR))

