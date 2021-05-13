import time

import numpy as np
from sklearn.linear_model import RANSACRegressor, HuberRegressor, TheilSenRegressor

from utils.attack_utilities import learn_model


def huber_model(x, y, epsilon_values, alpha):
    scores = []
    best_clf, best_count, best_time, best_score, = None, None, None, 0

    for eps in epsilon_values:
        t = time.process_time()
        clf = HuberRegressor(epsilon=eps, max_iter=1000, alpha=alpha)
        clf.fit(x, y)
        elapsed_time = time.process_time() - t
        score = clf.score(x[~clf.outliers_], y[~clf.outliers_])
        scores.append(score)
        if score > best_score:
            best_clf, best_eps, best_score, best_time = clf, eps, score, elapsed_time
    return best_clf, best_time


def ransac_model(x, y, model, count_values):
    scores = []
    best_clf, best_count, best_time, best_score, = None, None, None, 0

    for count in count_values:
        t = time.process_time()
        reg_ransac = RANSACRegressor(model, min_samples=count)
        reg_ransac.fit(x, y)
        elapsed_time = time.process_time() - t
        score = reg_ransac.score(x[reg_ransac.inlier_mask_], y[reg_ransac.inlier_mask_])
        scores.append(score)
        if score > best_score:
            best_clf, best_count, best_score, best_time = reg_ransac, count, score, elapsed_time
    return best_clf, best_time


def theilsen_model(x, y, n_subsamples):
    best_clf, best_breakdown = None, 0

    t = time.process_time()
    for samples_size in n_subsamples:
        reg_ts = TheilSenRegressor(n_subsamples=samples_size, copy_X=True)
        reg_ts.fit(x, y)
        breakdown = reg_ts.breakdown_
        if breakdown > best_breakdown:
            best_clf, best_breakdown = reg_ts, breakdown

    elapsed_time = time.process_time() - t
    return best_clf, elapsed_time


def trimclf(x, y, count, lam):
    length = x.shape[0]
    inds = np.random.permutation(length)[:count]
    inds = np.sort(inds.ravel())

    new_inds = np.array([])
    it = 0
    toterr = 10000
    last_error = 20000
    clf = learn_model(x, y, lam)

    while not np.array_equal(inds, new_inds) and it < 400 and last_error - toterr > 1e-5:
        new_inds = inds[:]
        last_error = toterr
        sub_x = x[inds, :]
        sub_y = y[inds]
        clf = clf.fit(sub_x, sub_y)
        predictions = clf.predict(x) - y
        resid_vec = np.square(predictions)

        # smallest residuals
        residual_indices = np.argsort(resid_vec, axis=0)[:count]
        smallest_n_resid = resid_vec[residual_indices].ravel()

        # set inds to indices of n largest values in error
        inds = np.sort(residual_indices.ravel())
        # recompute error
        toterr = sum(smallest_n_resid)
        it += 1
    return clf, lam, inds
