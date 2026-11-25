import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolve2d(f, h):
    """f: grayscale image (H x W), h: kernel (k x k, k odd)"""
    H, W = f.shape
    k = h.shape[0]
    pad = k // 2
    f_padded = np.pad(f, pad)          # zero padding around the image
    g = np.zeros((H, W))

    # TODO: your loops here
    # for each pixel (i, j): multiply the k x k patch of f_padded by h, sum, store in g[i, j]
    # remember: convolution = flip the kernel (np.flip(h)) before multiplying
    h_flipped = np.flip(h)
    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            window = f_padded[i - pad:i + pad + 1,j - pad:j + pad + 1]
            g[i - pad,j - pad] = (window * h_flipped).sum()           
    return g

def gaussian_kernel(sigma):
    pad = int(np.ceil(3 * sigma))
    k = 2 * pad + 1
    v = np.zeros(k)
    for i in range(k):
        v[i] = np.exp(-(((i-pad)**2)/(2 * (sigma ** 2))))
    h = np.outer(v,v)
    h /= h.sum()
    return h


img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE).astype(float)
h = gaussian_kernel(1)
np.set_printoptions(precision=3, suppress=True)
print(h)
print("sum:", h.sum())

mine = convolve2d(img, h)
k = h.shape[0]
ref = cv2.GaussianBlur(img, (k, k), 1, borderType=cv2.BORDER_CONSTANT)
print("max difference:", np.abs(mine - ref).max())

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("original")
ax[1].imshow(mine, cmap="gray"); ax[1].set_title("mine")
plt.show()