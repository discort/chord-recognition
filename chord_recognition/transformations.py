import numpy as np


TRAIN_MEAN = np.array([1.0620861 , 1.1210235 , 1.1626923 , 1.1714249 , 1.1911592 ,
       1.2532009 , 1.2950765 , 1.2884885 , 1.2528118 , 1.2003913 ,
       1.1907676 , 1.1697881 , 1.131795  , 1.1513147 , 1.1810027 ,
       1.1615353 , 1.1331217 , 1.1952394 , 1.2063102 , 1.1399081 ,
       1.1174718 , 1.1190281 , 1.1553446 , 1.1845721 , 1.1894056 ,
       1.1520061 , 1.1865602 , 1.209243  , 1.1375215 , 1.0698422 ,
       1.1039633 , 1.1174091 , 1.1045425 , 1.0806304 , 1.0407865 ,
       1.0460237 , 1.094718  , 1.1146975 , 1.030814  , 0.9873894 ,
       1.0504104 , 1.1573747 , 1.0813262 , 0.992421  , 1.0096711 ,
       1.0285554 , 1.0193492 , 1.0416723 , 0.99215835, 0.971497  ,
       1.0162044 , 1.0716246 , 0.9584235 , 0.89314705, 0.93678164,
       0.9933897 , 0.9318359 , 0.9319626 , 0.9164341 , 0.9076598 ,
       0.9118677 , 0.96250457, 0.88753295, 0.8403826 , 0.87852454,
       0.9691961 , 0.88115144, 0.8172574 , 0.8375757 , 0.8836306 ,
       0.8592457 , 0.87389505, 0.8106342 , 0.80556744, 0.81041896,
       0.878489  , 0.7980852 , 0.738269  , 0.7633766 , 0.84036475,
       0.77139986, 0.7485211 , 0.7490016 , 0.7626159 , 0.7475731 ,
       0.79209656, 0.73518264, 0.710283  , 0.72731626, 0.7909448 ,
       0.73633254, 0.71333   , 0.72848254, 0.75929815, 0.7250471 ,
       0.74173754, 0.7255356 , 0.7345132 , 0.73147273, 0.7641832 ,
       0.7151221 , 0.69239897, 0.69025   , 0.72472024, 0.6667584 ],
dtype=np.float32).reshape(1, -1)

TRAIN_STD = np.array([0.5669997 , 0.58026135, 0.58842736, 0.58474594, 0.5823268 ,
       0.5878695 , 0.579548  , 0.6032478 , 0.5904461 , 0.555948  ,
       0.56745136, 0.5634878 , 0.5533192 , 0.5448987 , 0.5594891 ,
       0.5297177 , 0.504992  , 0.5445396 , 0.5403601 , 0.5043022 ,
       0.5079558 , 0.522863  , 0.5230727 , 0.53380233, 0.48753622,
       0.47370377, 0.5363009 , 0.5393583 , 0.4965872 , 0.45686546,
       0.47417933, 0.50815755, 0.45596936, 0.4695911 , 0.4745014 ,
       0.4607748 , 0.45109326, 0.49029422, 0.4489223 , 0.44723687,
       0.460577  , 0.48810652, 0.4373333 , 0.43572068, 0.43508217,
       0.4516832 , 0.43802255, 0.45520446, 0.42085788, 0.43561447,
       0.43319827, 0.45437747, 0.41115174, 0.41076744, 0.42810234,
       0.44825003, 0.4183172 , 0.4256103 , 0.41929772, 0.43271577,
       0.43196324, 0.44969174, 0.40528798, 0.41065115, 0.4353304 ,
       0.4510096 , 0.401165  , 0.3990245 , 0.41738647, 0.43949497,
       0.39952007, 0.41190538, 0.3896694 , 0.39806995, 0.39894557,
       0.4186186 , 0.38257775, 0.37711626, 0.3895011 , 0.408666  ,
       0.36939684, 0.37088168, 0.36929625, 0.37637407, 0.37268773,
       0.3894435 , 0.36063802, 0.3613869 , 0.36560494, 0.38097677,
       0.35504884, 0.34727332, 0.35942587, 0.3716567 , 0.35516554,
       0.36109272, 0.35540017, 0.36029997, 0.3531999 , 0.36126   ,
       0.34777814, 0.3378683 , 0.33405286, 0.34044772, 0.3255972 ],
dtype=np.float32).reshape(1, -1)


def standardize(x, mean, std, eps=1e-20):
    """
    Rescale inputs to have a mean of 0 and std of 1
    """
    return (x - mean) / (std + eps)


class Rescale:
    """
    Rescale inputs to have a mean of 0 and std of 1
    It must speed up the convergence
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, frame):
        return standardize(frame, self.mean, self.std)
