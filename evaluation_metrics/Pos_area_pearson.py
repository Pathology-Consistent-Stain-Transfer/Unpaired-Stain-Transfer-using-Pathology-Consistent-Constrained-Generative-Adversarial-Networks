import Stain_seperation.stain_Norm_Vahadane as Norm_Vahadane
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import skimage.morphology as sm

def getImage(filepath):
    img_data = cv2.imread(filepath)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    return img_data


def Neg_area(img1_path,img0_path, size=750,step=750, re_fit=False, showfig = False):
    img0 = getImage(img0_path)
    img1 = getImage(img1_path)
    imgshape = img0.shape

    norm = Norm_Vahadane.normalizer()
    if re_fit:
        norm.fit([img0_path, img1_path])
    H0, E0 = norm.hematoxylin_eosin(img0)
    Neg0 = 1 - E0
    Neg0[Neg0 < 0.3] = 0
    Neg0[Neg0 >= 0.3] = 1
    kernel = sm.disk(5)
    Neg0 = sm.dilation(Neg0, kernel)
    Pos0 = H0 - Neg0
    Pos0[Pos0 < 0.1] = 1
    Pos0 = 1 - Pos0
    H1, E1 = norm.hematoxylin_eosin(img1)
    Neg1 = 1 - E1
    Neg1[Neg1 < 0.3] = 0
    Neg1[Neg1 >= 0.3] = 1
    Neg1 = sm.dilation(Neg1, kernel)
    Pos1 = H1 - Neg1
    Pos1[Pos1 <= 0.1] = 1
    Pos1 = 1 - Pos1

    Pos0[Pos0 < 0.2] = 0
    Pos1[Pos1 < 0.2] = 0

    if showfig:
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.imshow(img0)
        plt.subplot(2, 2, 2)
        plt.imshow(Pos0)
        plt.subplot(2, 2, 3)
        plt.imshow(img1)
        plt.subplot(2, 2, 4)
        plt.imshow(Pos1)
        plt.show()

    Pos0[Pos0 >= 0.2] = 1
    Pos1[Pos1 >= 0.2] = 1

    cv2.imwrite('4_4_0.png', Pos0*255)
    cv2.imwrite('4_4_1.png', Pos1*255)
    X_index = list(range(size, imgshape[0], step))
    X_index.append(imgshape[0])
    X_index = np.unique(np.asarray(X_index, dtype=np.int))
    Y_index = list(range(size, imgshape[1], step))
    Y_index.append(imgshape[1])
    Y_index = np.unique(np.asarray(Y_index, dtype=np.int))
    Pos0_area=[]
    Pos1_area=[]
    for x_id in X_index:
        for y_id in Y_index:
            Pos0_area.append(np.sum(Pos0[x_id - size:x_id, y_id - size:y_id]))
            Pos1_area.append(np.sum(Pos1[x_id - size:x_id, y_id - size:y_id]))
    '''
    Neg0_1 = Neg0[:, 0:imgshape[1]//2]
    Neg0_2 = Neg0[:, imgshape[1] // 2:imgshape[1]]
    Neg1_1 = Neg1[:, 0:imgshape[1] // 2]
    Neg1_2 = Neg1[:, imgshape[1] // 2:imgshape[1]]

    Neg0_1area = np.sum(Neg0_1[:])
    Neg0_2area = np.sum(Neg0_2[:])
    Neg1_1area = np.sum(Neg1_1[:])
    Neg1_2area = np.sum(Neg1_2[:])
    '''
    return Pos0_area, Pos1_area

if __name__ == '__main__':
    modelname = ['Cycle_GAN', 'Cycle_GAN_unet', 'Cycle_GAN_UnetSSIM', 'Cycle_GAN_SSIM',
                 'Cycle_GAN_pathology_cls',
                 'Cycle_GAN_pathology_seg', 'Cycle_GAN_pathology_seg_PN1_1',
                 'Cycle_GAN_pathology_seg_PN1_3', 'Cycle_GAN_pathology_seg_PN3_1',
                 'Cycle_GAN_PN_1_1','Cycle_GAN_PN_1_3', 'Cycle_GAN_PN_3_1',
                 ]
    model_id = 8
    print(model_id)
    epoch_id = 9
    wsi_name = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8]
    id = [1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 2, 3, 1, 2]
    # wsi_name = [4,4,4,4,4,7,7,7,7,7,7,7,7,7,7, 8,8,8,8,8,8,8,8,8,8, 9,9,9]
    # id = [4,1,2,3,4,1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3]
    xlsdata = {
        'fid': [],
        'Neg_area': [],
        'ref_Neg_area': []}
    for idx in range(wsi_name.__len__()):
        img1 = '/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/' + modelname[model_id] + '/exp3/epoch' + str(epoch_id) + '/output/A2B/WSI_data00' + str(wsi_name[idx]) + '/' + str(id[idx]).zfill(2) + '_pre_B.png'
        img0 = '/home/zhangbc/Mydataspace/LST/raw_data/neuroendocrine/Test/Ki67/WSI_data00' + str(wsi_name[idx])+'/'+str(id[idx]).zfill(2)+'.png'
        Neg0_area, Neg1_area = Neg_area(img1, img0, size=750, step=375, re_fit=True, showfig=False)
        print(str(wsi_name[idx]) + '_' + str(id[idx]))
        for pid in range(Neg0_area.__len__()):
            print('part', str(pid))
            print('reference ki67 Neg area:', Neg0_area[pid])
            print('generated ki67 Neg area:', Neg1_area[pid])
            xlsdata['fid'].append(str(wsi_name[idx]) + '_' + str(id[idx]) + '_part'+str(pid))
            xlsdata['ref_Neg_area'].append(Neg0_area[pid])
            xlsdata['Neg_area'].append(Neg1_area[pid])

    df = DataFrame(xlsdata)
    order = ['fid', 'Neg_area', 'ref_Neg_area']
    df = df[order]
    df.to_excel('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/' + modelname[model_id] + '/exp3/epoch' + str(epoch_id) + '/output/B_B_Pos.xlsx')