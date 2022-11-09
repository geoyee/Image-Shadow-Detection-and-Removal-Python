from typing import Tuple, Dict, Any
import numpy as np
import cv2
import pymeanshift as pms


def mean_shift(im: np.ndarray) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """使用meanshift分割图像"""
    im_luv = cv2.cvtColor(im, cv2.COLOR_BGR2Luv)
    _, seg_res, num_class = pms.segment(im_luv, spatial_radius=8, range_radius=5, min_density=200)
    regins : Dict[int, Dict[str, Any]] = dict()
    for i in range(num_class):
        regins[i] = dict()
        regins[i]["where"] = seg_res == i
    return seg_res, regins


def ext_feat(im: np.ndarray, regins: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """获取特征"""
    hsi = _RGB_to_HSI(im)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    for k, v in regins.items():
        area, center = _calc_center(regins[k]["where"])
        regins[k]["area"] = area
        regins[k]["center"] = center
        regins[k]["shadow"] = False
        regins[k]["refuse"] = False
        regins[k]["hsi"] = _calc_feat_from_roi(hsi, v["where"])
        regins[k]["hsv"] = _calc_feat_from_roi(hsv, v["where"])
        regins[k]["ycrcb"] = _calc_feat_from_roi(ycrcb, v["where"])
        regins[k]["Y"] = regins[k]["ycrcb"][0]
        regins[k]["R"] = regins[k]["hsi"][0] / (regins[k]["hsi"][-1] + 1e-12)
    return regins


def _calc_center(bim: np.ndarray) -> Tuple[int, np.ndarray]:
    bim = bim.astype("uint8") * 255
    contours, _ = cv2.findContours(bim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    center = np.array([x + w / 2, y + h / 2])
    return area, center


def _RGB_to_HSI(im_rgb: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(im_rgb)
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    eps = 1e-12
    im_i = (r + g + b) / 3
    min_rgb = np.zeros(r.shape, dtype=np.float32)
    min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
    min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
    min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
    im_s = 1 - 3 * min_rgb / (r + g + b + eps)
    num = ((r - g) + (r - b)) / 2
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + eps))
    im_h = np.where((b - g) > 0, 2 * np.pi - theta, theta)
    im_h = np.where(im_s == 0, 0, im_h)
    im_h /= (2 * np.pi)
    tmp_s = im_s - np.min(im_s)
    tmp_i = im_i - np.min(im_i)
    im_s = tmp_s / np.max(tmp_s)
    im_i = tmp_i / np.max(tmp_i)
    return cv2.merge([im_h, im_s, im_i])


def _calc_feat_from_roi(spf: np.ndarray, roi: np.ndarray):
    res = []
    for c in range(spf.shape[-1]):
        res.append(np.sum(spf[:, :, c] * roi) / (np.sum(spf) + 1e-12))
    return np.array(res)


# TEST
def img_debug(im, text="debug"):
    im = im.astype("uint8")
    dis_im = np.zeros([*im.shape, 3], "uint8")
    for i in np.unique(im):
        c = np.random.randint(256, size=(3, )).tolist()
        roi = (im == i).astype("uint8")
        dis_im[:, :, 2] += (int(c[0]) * roi)
        dis_im[:, :, 1] += (int(c[1]) * roi)
        dis_im[:, :, 0] += (int(c[2]) * roi)
    cv2.imshow(text, dis_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    im = cv2.imread("dataset/images/p21_1.jpg")
    # segmenting and detect
    seg_res, regins = mean_shift(im)
    # regins = ext_feat(im, regins)
    img_debug(seg_res)
    # print(len(regins), regins[1])
    