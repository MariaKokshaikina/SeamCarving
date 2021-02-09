import numpy as np
from functools import lru_cache


def seams_map_dp(img, energy):
    
    r, c, _ = img.shape

    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 1])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack


def seam_map_function_forward(img, energy):
    
    height = img.shape[0]
    width = img.shape[1]

    def CL(x,y):
        if x == y == 0:
            return 0
        elif y == 0:
            return 0
        elif x == 0:
            return energy[y][0][2] + energy[y-1][0][1]
        else:
            return energy[y][x-1][4] + energy[y-1][x][0]
    
    def CU(x,y):
        if x == y == 0:
            return 0
        elif y == 0:
            return energy[0][x-1][4]
        elif x == 0:
            return energy[y][0][2]
        else:
            return energy[y][x-1][4]

    def CR(x,y):
        if x == y == 0:
            return 0
        elif y == 0:
            return 0
        elif x == 0:
            return energy[y][0][2] + energy[y-1][0][3]
        else:
            return energy[y][x-1][4] + energy[y-1][x][3]

    M = np.empty((img.shape[0], img.shape[1]))
    backtrack = np.zeros_like(M, dtype=np.int)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if y == 0:
                M[y, x] = CU(x, 0)
            else:
                possible_come_from = np.asarray([
                    M[y-1,(x-1)% width] + CL(x,y) if x!=0 else np.inf,
                    M[y-1,x] + CU(x,y),
                    M[y-1,(x+1) % width] + CR(x,y) if x!=img.shape[1]-1 else np.inf,
                ])
                min_ = np.argmin(possible_come_from)
                backtrack[y,x] = x - 1 + min_
                M[y,x] = possible_come_from[min_]
    return M, backtrack


def carve_column_mask(img, m, backtrack):
    
    r,c,_ = img.shape
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(m[-1])
    
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]
    
    return mask

