import numpy as np
from PIL import Image
import utils

def e2c(e_img, face_w=256, mode='bilinear', cube_format='dice', sxy="sxy"):
    '''
    e_img:  ndarray in shape of [H, W, *]
    face_w: int, the length of each face of the cubemap
    '''
    assert len(e_img.shape) == 3
    h, w = e_img.shape[:2]
    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    xyz = utils.xyzcube(face_w)
    uv = utils.xyz2uv(xyz)
    coor_xy = utils.uv2coor(uv, h, w)

    cubemap = np.stack([
        utils.sample_equirec(e_img[..., i], coor_xy, order=order)
        for i in range(e_img.shape[2])
    ], axis=-1)

    if cube_format == 'horizon':
        pass
    elif cube_format == 'list':
        cubemap = utils.cube_h2list(cubemap)
    elif cube_format == 'dict':
        cubemap = utils.cube_h2dict(cubemap)
    elif cube_format == 'dice':
        cubemap = utils.cube_h2dice(cubemap, sxy)
    else:
        raise NotImplementedError()

    return cubemap


"""e_img = Image.open("deneme_0.png")
e_img = np.array(e_img)

kkk = e2c(e_img,sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)])
kkk = Image.fromarray(kkk)
kkk.save("denemekke.png")
"""