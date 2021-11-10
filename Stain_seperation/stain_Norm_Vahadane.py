"""
Stain normalization inspired by method of:

A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""
# windows: pip install spams-bin
# linux:pip install python-spams
import spams
import numpy as np
import Stain_seperation.stain_utils as ut


def get_stain_matrix(I, threshold=0.8, lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


class normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):

        self.stain_matrix_target = np.array([[0.62600721, 0.62330743, 0.46861798],
                                             [0.3203682, 0.5473311, 0.77317067]])
        # Ki67 Normalization initial matirx obtained from "Sample_target"
        # [[0.58594418, 0.68469766, 0.43342651]
        #  [0.3203682, 0.5473311, 0.77317067]]

        # [[0.62600721,0.62330743,0.46861798],
        #  [0.35395456,0.58236586,0.73182387]]

        # [[0.58583788, 0.66078505, 0.46920901],
        #  [0.3536072, 0.56354522, 0.74657801]]

        # HE Normalization initial matirx obtained from "Sample_target"
        # self.stain_matrix_target = np.array([[0.60559458, 0.69559906, 0.38651928],
        #                                      [0.1100605, 0.94701408, 0.30174662]])
        # [[0.59958405,0.70248408,0.38342546]
        #  [0.06893222,0.95236792,0.2970584]]

        # [[0.60559458 0.69559906 0.38651928]
        #  [0.1100605  0.94701408 0.30174662]]

        # [[0.60715608 0.72015621 0.3357626]
        #  [0.21154943 0.9271104  0.30937542]]

    def fit(self, target_list):
        if target_list.__len__() > 1:
            Ws = []
            for f_id in range(target_list.__len__()):
                target = ut.read_image(target_list[f_id])
                target = ut.standardize_brightness(target)
                stain_matrix_target = get_stain_matrix(target)
                Ws.append(stain_matrix_target)
            Ws = np.asarray(Ws)
            Median_W = np.median(Ws, axis=0)
            self.stain_matrix_target = ut.normalize_rows(Median_W)
            print('WSI target stain matrix: ', self.stain_matrix_target)
        else:
            target = ut.read_image(target_list[0])
            target = ut.standardize_brightness(target)
            self.stain_matrix_target = get_stain_matrix(target)
            print('Single target image stain matrix: ', self.stain_matrix_target)

    def stains_Vec_RGB(self, stain_matrix_target):
        return ut.OD_to_RGB(stain_matrix_target)

    def transform(self, I):
        I = ut.standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)

    def hematoxylin_eosin(self, I):
        I = ut.standardize_brightness(I)
        h, w, _ = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)

        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)

        E = source_concentrations[:, 1].reshape(h, w)
        E = np.exp(-1 * E)

        # H = np.reshape(source_concentrations[:, 0], newshape=(h*w, 1))
        # H = (255 * np.exp(-1 * np.dot(H, np.reshape(stain_matrix_source[0],
        #                                             newshape=(1, 3))).reshape(I.shape))).astype(np.uint8)
        # E = np.reshape(source_concentrations[:, 1], newshape=(h*w, 1))
        # E = (255 * np.exp(-1 * np.dot(E, np.reshape(stain_matrix_source[1],
        #                                             newshape=(1, 3))).reshape(I.shape))).astype(np.uint8)
        return H, E
