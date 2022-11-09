from typing import Tuple
import numpy as np
import cv2
import pymeanshift as pms


def meanShift(im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    im_luv = cv2.cvtColor(im, cv2.COLOR_BGR2Luv)
    dummy, seg, _ = pms.segment(im_luv, spatial_radius=9, range_radius=15, min_density=200)
    return dummy, seg


# TEST
def imgDebug(im, text="debug"):
    im = im.astype("uint8") * 10
    im = cv2.applyColorMap(im, cv2.COLORMAP_HSV)
    cv2.imshow(text, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    im = cv2.imread("dataset/images/lssd577.jpg")
    dummy, seg = meanShift(im)
    seg += 1
    seg_num = np.max(np.unique(seg))
    imgDebug(seg)
    