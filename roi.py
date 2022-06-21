from typing import Any

import numpy as np

# We'll use a linear combination (matrix multiplication) of some Points of Intrest of the mesh to determine the RoI.
# Here we define those Points of Intrest and the matrix to combine them.

forehead_PoI = (103,  67, 109,  10, 338, 297, 332,
                333, 299, 337, 151, 108,  69, 104)

forehead_comb = 1.4 * np.eye(len(forehead_PoI)) - .4 * np.eye(len(forehead_PoI))[:, ::-1]


hand_PoI = (0, 1, 5, 9, 13, 17)

hand_comb = [[.7, .0,  .0,  .0,  .0,  .3],
             [.0, .8, .05, .05, .05, .05],
             [.0, .0,  1.,  .0,  .0,  .0],
             [.0, .0,  .0,  1.,  .0,  .0],
             [.0, .0,  .0,  .0,  1.,  .0],
             [.0, .0,  .0,  .0,  .0,  1.]]


face_RoI_ = (152, 148, 176, 149, 150, 136, 172,  58, 215, 132, 147, 93, 123, 116, 127, 139,  21,  54, 103,  67,
             109, 10, 338, 297, 332, 284, 251, 389, 264, 345, 352, 376, 433, 367, 397, 365, 379, 378, 400, 377)

face_RoI = (152, 148, 176, 149, 169, 135, 138, 215, 177, 137, 227, 34, 139, 71, 68, 103, 67, 109, 10,
            338, 297, 332, 298, 301, 368, 264, 447, 366, 401, 435, 367, 367, 364, 394, 378, 400, 377)

left_eye_RoI = (463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341)

right_eye_RoI = (130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25)

right_eyebrow_PoI = (55, 65, 52, 63, 105, 66, 107)

left_eyebrow_PoI = (336, 296, 334, 293, 282, 295, 285)

mouth_RoI = (0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37)

excludes = (left_eye_RoI, right_eye_RoI, left_eyebrow_PoI, right_eyebrow_PoI, mouth_RoI)

forehead_small_PoI = (66, 107,   69,  108,  67, 109, 296, 336, 299, 337, 297, 338)

forehead_small_comb = [[.6,  .2,   .1,   .1,   0,   0,   0,  0,    0,    0,  0,  0],
                       [ 0,   0, -.15, -.15,  .5,  .8,   0,  0,    0,    0,  0,  0],
                       [ 0,   0,    0,    0,   0,   0,   0,  0, -.15, -.15, .5, .8],
                       [ 0,   0,    0,    0,   0,   0,  .6, .2,   .1,   .1,  0,  0],]

left_under_eye_PoI = (329, 371, 423, 426, 427, 433, 376, 352, 345, 346, 347, 348)  # (329, 371, 358?, 423, 426/58, 427, 411, 376, 352, 345, 346, 347, 348, 329)

right_under_eye_PoI = (147, 123, 116, 117, 118, 119, 100, 142, 203, 206, 207, 213)  #, 329, 371, 423, 426, 427, 433, 376, 352, 345, 346, 347, 348)

nose_PoI = (55, 8, 285, 412, 420, 279, 327, 326, 2, 97, 98, 49, 198, 188)


nose_tip_PoI = (218, 198, 420, 438)


upper_lip = (287, 57, 206, 426)
lower_lip = (287, 57, 211, 431)


chin = (418, 194, 176, 400)

left_down_check = (436, 433, 364, 394, 431)
right_down_check = (216, 213, 135, 169, 211)

