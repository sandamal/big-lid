import numpy as np
from scipy.spatial.distance import cdist


def get_lids(X, k=10, batch=None):
    if batch is None:
        lid_batch = mle_batch(X, None, k=k)
    else:
        lid_batch = mle_batch(X, batch, k=k)

    lids = np.asarray(lid_batch, dtype=np.float32)
    return lids


# lid of a batch of query points X
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))

    if batch is None:
        a = cdist(data, data)
        # get the closest k neighbours
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    else:
        batch = np.asarray(batch, dtype=np.float32)
        a = cdist(data, batch)
        # get the closest k neighbours
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 0:k]

    a = np.apply_along_axis(f, axis=1, arr=a)
    return a
