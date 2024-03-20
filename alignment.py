import math
import random
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates homography from image 1 to image 2 from matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows, num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt
        A[i*2] = [a_x, a_y, 1, 0, 0, 0, -b_x*a_x, -b_x*a_y, -b_x]
        A[(i*2)+1] = [0, 0, 0, a_x, a_y, 1, -b_y*a_x, -b_y*a_y, -b_y]

    _, _, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    # s is a 1-D array of singular values sorted in descending order
    # U, Vt are unitary matrices
    # Rows of Vt are the eigenvectors of A^TA.
    # Columns of U are the eigenvectors of AA^T.

    # Homography to be calculated
    H = np.eye(3)
    H = Vt[-1].reshape(3, 3)
    return H


def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    k = 1 if m == eTranslate else 4 if m == eHomography else 0
    maxInliers, mInliners = 0, []
    for _ in range(nRANSAC):
        ms = random.sample(matches, k)
        H = np.eye(3, 3)
        if m == eHomography:
            H = computeHomography(f1, f2, ms)
        else:
            H[0, 2] = f2[ms[0].trainIdx].pt[0] - f1[ms[0].queryIdx].pt[0]
            H[1, 2] = f2[ms[0].trainIdx].pt[1] - f1[ms[0].queryIdx].pt[1]
        inliers = getInliers(f1, f2, matches, H, RANSACthresh)
        if len(inliers) > maxInliers:
            maxInliers = len(inliers)
            mInliners = inliers
    M = leastSquaresFit(f1, f2, matches, m, mInliners)
    return M


def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        # Determines if the ith matched feature f1[id1], when transformed
        # by M, is within RANSACthresh of its match in f2.
        # If so, appends i to inliers
        qt = M.dot(np.array([f1[matches[i].queryIdx].pt[0],
                             f1[matches[i].queryIdx].pt[1], 1]).T)
        x1, y1 = [qt[0]/qt[2], qt[1]/qt[2]]
        x2, y2 = np.array(f2[matches[i].trainIdx].pt)[:2]
        if math.sqrt((x2 - x1)**2 + (y2 - y1)**2) <= RANSACthresh:
            inlier_indices.append(i)
    return inlier_indices


def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    M = np.eye(3)

    if m == eTranslate:
        # For spherically warped images, the transformation is a
        # translation and only has two degrees of freedom.
        # Therefore, we simply compute the average translation vector
        # between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            # Computes the average translation vector
            # over all inliers.
            p1 = f1[matches[inlier_indices[i]].queryIdx].pt
            p2 = f2[matches[inlier_indices[i]].trainIdx].pt
            u += p2[0] - p1[0]
            v += p2[1] - p1[1]
        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0, 2] = u
        M[1, 2] = v

    elif m == eHomography:
        # Computes a homography M using all inliers.
        inliers = []
        for i in inlier_indices:
            inliers.append(matches[i])
        M = computeHomography(f1, f2, inliers)
    else:
        raise Exception("Error: Invalid motion model.")
    return M
