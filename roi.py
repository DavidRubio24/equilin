from typing import Any

import numpy as np

# We'll use a linear combination (matrix multiplication) of some Points of Intrest of the mesh to determine the RoI.
# Here we define those Points of Intrest and the matrix to combine them.

forehead_PoI = (103,  67, 109,  10, 338, 297, 332,
                333, 299, 337, 151, 108,  69, 104)

forehead_comb = 1.4 * np.eye(len(forehead_PoI)) - .4 * np.eye(len(forehead_PoI))[:, ::-1]


hand_PoI = (0, 1, 5, 9, 13, 17)

hand_comb = [[.7,  0., 0., 0., 0., 0.],
             [0.,  .8, 0., 0., 0., 0.],
             [0., .05, 1., 0., 0., 0.],
             [0., .05, 0., 1., 0., 0.],
             [0., .05, 0., 0., 1., 0.],
             [.3, .05, 0., 0., 0., 1.]]
