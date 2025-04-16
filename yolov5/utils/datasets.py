import cv2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # resize ratio
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # width, height
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    ratio = r, r
    pad = (dw, dh)
    return im, ratio, pad
