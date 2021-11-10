import cv2
import numpy as np
from sewar.full_ref import ssim, msssim, psnr
from pandas import DataFrame
from skimage.measure import compare_ssim

modelname = ['Cycle_GAN', 'Cycle_GAN_unet', 'Cycle_GAN_UnetSSIM', 'Cycle_GAN_SSIM', 'Cycle_GAN_pathology_cls', 'Cycle_GAN_pathology_seg']
model_id = 2
epoch_id = 5
print(modelname[model_id], epoch_id)
wsi_name = [1,1,2,2,2,3,3,4,4,4,4,4,5,5,6,6,7,7,7,8,8]
id = [1,2,1,2,3,1,2,1,2,3,4,5,1,2,1,2,1,2,3,1,2]

xlsdata = {
    'fid': [],
    'cs': [],
    'ssim': []}

# wsi_name = [1, 1]
# id = [1, 2]
for idx in range(wsi_name.__len__()):
    img0 = cv2.imread('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/'+modelname[model_id]+'/exp1/epoch'+str(epoch_id)+'/output/A2B/WSI_data00'+str(wsi_name[idx])+'/0'+str(id[idx])+'_real_A.png')
    img1 = cv2.imread('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/'+modelname[model_id]+'/exp1/epoch'+str(epoch_id)+'/output/A2B/WSI_data00'+str(wsi_name[idx])+'/0'+str(id[idx])+'_pre_B.png')

    tile0 = img0[0:1500, 0:1500, :]
    tile1 = img1[0:1500, 0:1500, :]

    degree_ssim1, degree_ssim2 = ssim(tile0, tile1, ws=5)

    print('<-----WSI_data00'+str(wsi_name[idx])+'/0'+str(id[idx])+'------>')
    print('>>part 1')
    #print('msssim:', degree_msssim.real*100)
    print('ssim1:', degree_ssim1 * 100)
    print('ssim2:', degree_ssim2 * 100)
    xlsdata['fid'].append(str(wsi_name[idx])+'_'+str(id[idx])+'part1')
    xlsdata['cs'].append(degree_ssim2*100)
    xlsdata['ssim'].append(degree_ssim1*100)

    tile0 = img0[0:1500, 1505:3010, :]
    tile1 = img1[0:1500, 1505:3010, :]

    degree_ssim1, degree_ssim2 = ssim(tile0, tile1, ws=5)

    print('>>part 2')
    print('ssim1:', degree_ssim1 * 100)
    print('ssim2:', degree_ssim2 * 100)
    print('<---------end---------->')
    xlsdata['fid'].append(str(wsi_name[idx])+'_'+str(id[idx])+'part2')
    xlsdata['ssim'].append(degree_ssim1*100)
    xlsdata['cs'].append(degree_ssim2 * 100)

df = DataFrame(xlsdata)
df.to_excel('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/'+modelname[model_id]+'/exp1/epoch'+str(epoch_id)+'/output/B_B_SSIM.xlsx')
