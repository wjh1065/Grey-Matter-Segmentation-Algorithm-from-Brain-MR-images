import os
import matplotlib.pyplot as plt
from skimage import exposure
import nibabel as nib
from numpy import percentile

def get_data_list_input(root):
    dir_list = os.listdir(root)
    pet_list = []
    for i in dir_list:
        if i.startswith('ADNI'): # name
            print('------job Start------')
            data_path = os.path.join(root, i)
            subject_rPET = (data_path + '/thr_nan_T1.nii.gz')
            print('subject : ', subject_rPET)
            subject_img_header_rPET = nib.load(subject_rPET)

            subject_img_rPET = subject_img_header_rPET.get_fdata()
            data_rPET_1 = subject_img_header_rPET.get_fdata()

            data_rPET = data_rPET_1[data_rPET_1 > 0]


            q99 = percentile(data_rPET, 99)
            upper = q99
            print('upper :  ', upper)

            data_graph = data_rPET[data_rPET < upper]
            # plot basic intensity graph
            hist1, bins_center1 = exposure.histogram(data_graph)
            plt.plot(bins_center1, hist1, lw=2)

            plt.tight_layout()
            plt.show()

            # step 2. make thr lower upper threshold
            threshold_indices = subject_img_rPET <= 0
            subject_img_rPET[threshold_indices] = 0
            threshold_indices = subject_img_rPET >= upper
            subject_img_rPET[threshold_indices] = upper


            final_nii = nib.Nifti1Image(subject_img_rPET, subject_img_header_rPET.affine)
            final_nii.to_filename(os.path.join(data_path, 'Input_T1.nii.gz'))

            print('------job End------')
            print('-------------------')
    return pet_list

root = '/home/wjh1065/Desktop/3090_Vnet/Vnet/data/thr2/data2'
pet_list = get_data_list_input(root)