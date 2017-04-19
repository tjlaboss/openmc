from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import scipy.sparse as sps
from copy import deepcopy

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def block_diag(array):

    ni, ng, ng = array.shape
    diags = np.zeros((ng*2-1, ni * ng))
    ndiag = [0] + [g for g in range(1,ng)] + [-g for g in range(1,ng)]

    for i in range(ni):
        for r in range(ng):
            for c in range(ng):
                diags[c-r][i*ng+r] = array[i, r, c]

    diags2 = [diags[0]]
    for g in range(1,ng):
        diags2.append(diags[g][:-g])
    for g in range(1,ng):
        diags2.append(diags[-g][g:])

    return sps.diags(diags2, ndiag)

def map_array(array, to_shape, normalize=True, lcm_applied=False):

    from_shape = array.shape
    num_dims = len(from_shape)
    if len(to_shape) != num_dims:
        msg = 'from_shape and to_shape have different dimension!'
        raise ValueError(msg)

    if not lcm_applied:
        lcm_shape = []
        for d in range(num_dims):
            lcm_shape.append(lcm(from_shape[d], to_shape[d]))
        lcm_shape = tuple(lcm_shape)

        # Map the array to the lcm mesh
        array = map_array(array, lcm_shape, normalize, True)
        from_shape = lcm_shape

    # loop over dimensions
    for d in range(num_dims):

        # condense along dimension
        if from_shape[d] > to_shape[d]:
            ratio = from_shape[d] / to_shape[d]
            block = (1,) * d + (ratio,) + (1,) * (num_dims-1-d)
            axes = tuple(range(num_dims,2*num_dims))
            if normalize:
                array = block_view(array, block).mean(axis=axes)
            else:
                array = block_view(array, block).sum(axis=axes)

        # expand along dimension
        else:
            ratio = to_shape[d] / from_shape[d]
            if normalize:
                array = np.repeat(array, ratio, axis=d)
            else:
                array = np.repeat(array, ratio, axis=d) / float(ratio)

    return array

def block_view(A, block):
    """Provide a block view to 3+D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""

    block_dim = len(block)

    # Check that blocks fit nicely into first dimensions of A
    if block_dim > len(A.shape):
        msg = 'Cannot access block_view of {}-D array with block size {}-D'\
              ' that is greater than array dimensions'.format(len(A.shape),
                                                              block_dim)
        raise ValueError(msg)
    else:
        block = block + A.shape[block_dim:]
        block_dim = len(A.shape)

    for i,b in enumerate(block):
        if A.shape[i] % b != 0:
            msg = 'Cannot access block_view of {}-D array with block {}'\
                  ' since block dimension {} does not fit evenly in'\
                  ' Array dimension {}'.format(len(A.shape), block,
                                               b, A.shape[i])
            raise ValueError(msg)

    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = ()
    strides = ()

    for i,b in enumerate(block):
        shape = shape + (A.shape[i] / block[i],)
        strides = strides + (b * A.strides[i],)

    shape = shape + block
    strides = strides + A.strides

    return ast(A, shape=shape, strides=strides)

def nan_inf_to_zero(array):

    array[array == -np.inf] = 0.
    array[array ==  np.inf] = 0.
    return np.nan_to_num(array)

def nan_inf_to_one(array):

    array[array == -np.inf] = 1.0
    array[array ==  np.inf] = 1.0
    array[array ==  np.nan] = 1.0
    return array

