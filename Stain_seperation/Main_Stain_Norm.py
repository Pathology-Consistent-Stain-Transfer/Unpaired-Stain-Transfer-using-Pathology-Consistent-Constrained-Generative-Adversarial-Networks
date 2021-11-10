import time, cv2, os
import Stain_seperation.stain_utils as utils
import Stain_seperation.stain_Norm_Vahadane as Norm_Vahadane
import glob

def assure_path_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
                raise


def stain_Norm(target_tiles_list, source_tiles_list, output_path, re_fit=False):
    assure_path_exists(output_path)
    if target_tiles_list.__len__() > 0 and source_tiles_list.__len__() > 0:
        start_time = time.time()
        print('Start stain normalization')
        norm = Norm_Vahadane.normalizer()
        if re_fit:
            norm.fit(target_tiles_list)
        print('done: target stain matrix')
        for counter, img in enumerate(source_tiles_list):
            img_name = (img.split('/')[-1])
            transformed_img = norm.transform(utils.read_image(img))
            cv2.imwrite(os.path.join(output_path, img_name), cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
        elapsed = (time.time() - start_time)
        print("--- %s seconds ---" % round((elapsed / 2), 2))
    else:
        print('the soucer or target path is invalidate')


if __name__ == '__main__':

    source_tiles_path = '/home/zhangbc/Mydataspace/LST/raw_data/GLAS/Train/img'
    target_tiles_path = '/home/zhangbc/Mydataspace/LST/raw_data/NewDataset_x20/Expert/'
    Norm_output_path = '/home/zhangbc/Mydataspace/LST/raw_data/GLAS/Train_Norm/img'

    target_tiles_list = sorted(glob.glob(os.path.join(target_tiles_path, 'HE') + '/*/*.png'))
    source_tiles_list = sorted(glob.glob(os.path.join(source_tiles_path, '*.bmp')))
    stain_Norm(target_tiles_list, source_tiles_list, Norm_output_path, re_fit=True)