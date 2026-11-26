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

def b_kernel(window, sigma_r):
    k = window.shape[0]
    center = window[k // 2, k // 2]
    A = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            A[i,j] = np.exp(-((window[i,j] - center) ** 2) / (2 * (sigma_r ** 2)))
    return A     


def bilateral_filter(f, sigma_s, sigma_r):
    H,W = f.shape
    ker_s = gaussian_kernel(sigma_s)
    k = ker_s.shape[0]
    pad = k // 2
    f_padded = np.pad(f, pad)
    g = np.zeros((H, W))
    for i in range(pad, H + pad):
        for j in range(pad, W + pad):
            window = f_padded[i - pad:i + pad + 1,j - pad:j + pad + 1]
            ker_b = b_kernel(window, sigma_r)
            g[i-pad, j-pad] = (window * ker_s * ker_b).sum()
            w_p = (ker_s * ker_b).sum()
            g[i-pad, j-pad] /= w_p
    return g



img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE).astype(float)

# lesson 1
# h = gaussian_kernel(1)
# np.set_printoptions(precision=3, suppress=True)
# print(h)
# print("sum:", h.sum())

# mine = convolve2d(img, h)
# k = h.shape[0]
# ref = cv2.GaussianBlur(img, (k, k), 1, borderType=cv2.BORDER_CONSTANT)
# print("max difference:", np.abs(mine - ref).max())

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].imshow(img, cmap="gray"); ax[0].set_title("original")
# ax[1].imshow(mine, cmap="gray"); ax[1].set_title("mine")
# plt.show()

# lesson 2
img = cv2.resize(img, None, fx=0.25, fy=0.25)

# test 1: huge sigma_r should equal plain Gaussian
bil = bilateral_filter(img, 2, 1e6)
gau = convolve2d(img, gaussian_kernel(2))
print("max difference:", np.abs(bil - gau).max())

# test 2: visual comparison
bil = bilateral_filter(img, 2, 30)
ocv = cv2.bilateralFilter(img.astype(np.float32), 13, 30, 2)

fig, ax = plt.subplots(1, 4, figsize=(16, 5))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("original")
ax[1].imshow(gau, cmap="gray"); ax[1].set_title("gaussian")
ax[2].imshow(bil, cmap="gray"); ax[2].set_title("bilateral (mine)")
ax[3].imshow(ocv, cmap="gray"); ax[3].set_title("bilateral (OpenCV)")
plt.show()
