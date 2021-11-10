import cv2
import numpy as np
from sewar.full_ref import ssim, msssim, psnr
from pandas import DataFrame

modelname = ['Cycle_GAN', 'Cycle_GAN_unet', 'Cycle_GAN_UnetSSIM', 'Cycle_GAN_SSIM', 'Cycle_GAN_pathology_cls', 'Cycle_GAN_pathology_seg']
model_id = 3
epoch_id = 9
print(modelname[model_id], epoch_id)
wsi_name = [1,1,2,2,2,3,3,4,4,4,4,4,5,5,6,6,7,7,7,8,8]
id = [1,2,1,2,3,1,2,1,2,3,4,5,1,2,1,2,1,2,3,1,2]
xlsdata = {
    'fid': [],
    'msssim': [],
    'ssim': [],
    'psnr': [],
    'mae': []}

# wsi_name = [1, 1]
# id = [1, 2]
for idx in range(wsi_name.__len__()):
    img0 = cv2.imread('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/'+modelname[model_id]+'/exp1/epoch'+str(epoch_id)+'/output/A2B/WSI_data00'+str(wsi_name[idx])+'/0'+str(id[idx])+'_real_A.png')
    img1 = cv2.imread('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/'+modelname[model_id]+'/exp1/epoch'+str(epoch_id)+'/output/A2B/WSI_data00'+str(wsi_name[idx])+'/0'+str(id[idx])+'_rec_A.png')

    tile0 = img0[0:1500, 0:1505, :]
    tile1 = img1[0:1500, 0:1505, :]
    degree_msssim = msssim(tile0, tile1, ws=16)
    degree_psnr = psnr(tile0, tile1)
    degree_ssim, _ = ssim(tile0, tile1, ws=16)
    tile0_gray = np.asarray(cv2.cvtColor(tile0, cv2.COLOR_BGR2GRAY), dtype=float)
    tile1_gray = np.asarray(cv2.cvtColor(tile1, cv2.COLOR_BGR2GRAY), dtype=float)
    diff_img = np.abs(tile0_gray-tile1_gray)
    mae = np.mean(diff_img, axis=(0, 1))
    print('<-----WSI_data00'+str(wsi_name[idx])+'/0'+str(id[idx])+'------>')
    print('>>part 1')
    print('mae:', mae)
    print('msssim:', degree_msssim.real*100)
    print('ssim:', degree_ssim*100)
    print('psnr:', degree_psnr)
    xlsdata['fid'].append(str(wsi_name[idx])+'_'+str(id[idx])+'part1')
    xlsdata['msssim'].append(degree_msssim.real*100)
    xlsdata['ssim'].append(degree_ssim*100)
    xlsdata['psnr'].append(degree_psnr)
    xlsdata['mae'].append(mae)

    tile0 = img0[0:1500, 1505:3010, :]
    tile1 = img1[0:1500, 1505:3010, :]

    degree_msssim = msssim(tile0, tile1, ws=16)
    degree_psnr = psnr(tile0, tile1)
    degree_ssim, _ = ssim(tile0, tile1, ws=16)
    tile0_gray = np.asarray(cv2.cvtColor(tile0, cv2.COLOR_BGR2GRAY), dtype=float)
    tile1_gray = np.asarray(cv2.cvtColor(tile1, cv2.COLOR_BGR2GRAY), dtype=float)
    diff_img = np.abs(tile0_gray-tile1_gray)
    mae = np.mean(diff_img, axis=(0, 1))
    print('>>part 2')
    print('mae:', mae)
    print('msssim:', degree_msssim.real*100)
    print('ssim:', degree_ssim*100)
    print('psnr:', degree_psnr)
    print('<---------end---------->')
    xlsdata['fid'].append(str(wsi_name[idx])+'_'+str(id[idx])+'part2')
    xlsdata['msssim'].append(degree_msssim.real*100)
    xlsdata['ssim'].append(degree_ssim*100)
    xlsdata['psnr'].append(degree_psnr)
    xlsdata['mae'].append(mae)
df = DataFrame(xlsdata)
df.to_excel('/home/zhangbc/Mydataspace/LST/neuroendocrine/mymodelx288/'+modelname[model_id]+'/exp1/epoch'+str(epoch_id)+'/output/A_A.xlsx')
