import math
import cv2
import numpy as np


class ImageInfo:  # this is what constitutes ipv
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    h = img.shape[0] - 1
    w = img.shape[1] - 1

    corners = np.array([[0, 0, 1],   # Top-left corner
                        [0, w, 1],   # Top-right corner
                        [h, 0, 1],   # Bottom-left corner
                        [h, w, 1]    # Bottom-right corner
                        ])

    corners = np.matmul(M, corners.T).T
    corners = corners / corners[:, -1:]  # normalize
    topLeft, topRight, bottomLeft, bottomRight = corners

    minX = min(topLeft[0], bottomLeft[0], topRight[0], bottomRight[0])
    minY = min(topLeft[1], bottomLeft[1], topRight[1], bottomRight[1])
    maxX = max(topLeft[0], bottomLeft[0], topRight[0], bottomRight[0])
    maxY = max(topLeft[1], bottomLeft[1], topRight[1], bottomRight[1])

    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """

    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    blendWidth = maxX - minX if maxX - minX < 2*blendWidth else blendWidth
    a = np.concatenate((np.linspace(0.0, 1.0, blendWidth),
                        np.ones(maxX - minX - 2*blendWidth),
                        np.linspace(1.0, 0.0, blendWidth)))
    src = np.ones((img.shape[0], img.shape[1], 4))
    src[:, :, 0] = img[:, :, 0]
    src[:, :, 1] = img[:, :, 1]
    src[:, :, 2] = img[:, :, 2]
    warp_src = cv2.warpPerspective(src, np.linalg.inv(M),
                                   (acc.shape[1], acc.shape[0]),
                                   flags=(cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST))

    for i in range(minX, maxX):
        warp_src[:, i, :3] = warp_src[:, i, :3] * a[i - minX]
        warp_src[:, i, 3] = np.full((warp_src.shape[0]), a[i - minX])
        for j in range(minY, maxY):
            if np.array_equal(warp_src[j, i, :3], [0, 0, 0]):
                warp_src[j, i, 3] = 0.0
            acc[j, i] += warp_src[j, i]


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    img = np.zeros((acc.shape[0], acc.shape[1], 3), dtype=np.uint8)
    for r in range(acc.shape[0]):
        for c in range(acc.shape[1]):
            if acc[r, c, 3] > 0:
                img[r, c] = (acc[r, c, 0:3] / acc[r, c, 3]).astype(int)
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the
       accumulated image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img)
         and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all
         tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all
         tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same
         width)
         translation: transformation matrix so that top-left corner of
         accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        bounds = imageBoundingBox(img, M)
        minX = min(minX, bounds[0])
        minY = min(minY, bounds[1])
        maxX = max(maxX, bounds[2])
        maxY = max(maxY, bounds[3])

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # Trims left edge and to take out the vertical drift if it's a 360 panorama
    # Shifts it left by the correct amount, then handles the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
