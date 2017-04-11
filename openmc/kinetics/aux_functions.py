from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import scipy.sparse as sps
from copy import deepcopy

def block_view2(A, block):
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


def map_array(array, to_shape, normalize=True):

    from_shape = array.shape
    array = deepcopy(array)
    num_dims = len(from_shape)

    # loop over dimensions
    for d in range(num_dims):

        # condense along dimension
        if from_shape[d] > to_shape[d]:
            ratio = from_shape[d] / to_shape[d]
            block = (1,) * d + (ratio,) + (1,) * (num_dims-1-d)
            axes = tuple(range(num_dims,2*num_dims))
            if normalize:
                array = block_view2(array, block).mean(axis=axes)
            else:
                array = block_view2(array, block).sum(axis=axes)

        # expand along dimension
        else:
            ratio = to_shape[d] / from_shape[d]
            if normalize:
                array = np.repeat(array, ratio, axis=d)
            else:
                array = np.repeat(array, ratio, axis=d) / ratio

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


def get_flux(amp_shape, mesh_shape, ng=2):

    # Translate the amplitude to the mesh of interest
    amplitude = np.random.random(amp_shape + (ng,))
    amplitude.shape = (amp_shape[0], amp_shape[1], amp_shape[2], ng)

    for d in range(3):

        # condense along dimension
        if amp_shape[d] > mesh_shape[d]:
            ratio = amp_shape[d] / mesh_shape[d]
            block = (1,) * d + (ratio,) + (1,) * (3-d)
            amplitude = block_view(amplitude, block).mean(axis=(4,5,6,7))

        # expand along dimension
        else:
            ratio = mesh_shape[d] / amp_shape[d]
            tile = (1,) * d + (ratio,) + (1,) * (3-d)
            amplitude = np.tile(amplitude, tile)

    # reshape and multiply by the shape
    print(amplitude.shape)
    amplitude.shape = (np.prod(mesh_shape[:3]), ng)
    return amplitude
