#!/usr/bin/env python3
"""slicing """


def np_slice(matrix, axes={}):
    """slicing"""
    x, y, z = None, None, None
    slice_xyz = []
    for xx in range(matrix.ndim):
        if xx in axes.keys():
            slc = axes[xx]
            a, b, c = slc[0], None, None
            if len(slc) == 2:
                b = slc[1]
            if len(slc) == 3:
                c = slc[2]
            if matrix.ndim == 3:
                slice_xyz.append(slice(b, a, c))
            else:
                slice_xyz.append(slice(a, b, c))
        else:
            slice_xyz.append(slice(None, None, None))
    return matrix[slice_xyz]
