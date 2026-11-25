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
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            window = f_padded[i-1:i+2,j-1:j+2]
            g[i - 1,j - 1] = (window * h_flipped).sum()           
    return g

img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE).astype(float)
h = np.ones((3, 3)) / 9

mine = convolve2d(img, h)
ref = cv2.filter2D(img, -1, np.flip(h), borderType=cv2.BORDER_CONSTANT)
print("max difference:", np.abs(mine - ref).max())

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("original")
ax[1].imshow(mine, cmap="gray"); ax[1].set_title("mine")
plt.show()