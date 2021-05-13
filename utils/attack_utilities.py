import numpy as np
import sympy as sy
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures


def get_centroids(data, centroid_num):
    idx = np.random.randint(np.size(data, axis=0), size=centroid_num)
    return data[idx, :]


class RbfFeatures:
    def __init__(self, num_centroids, random_state):
        self.num_centroids = num_centroids
        self.sigma_sq = None
        self.centroids = None
        self.random_state = random_state

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, data):
        np.random.seed(self.random_state)
        self.sigma_sq = np.median(cdist(data, data, 'sqeuclidean'))
        self.centroids = get_centroids(data, self.num_centroids)
        return self

    def transform(self, x):
        pairwise_sq_dists = cdist(x, self.centroids, 'sqeuclidean')
        return np.exp(-pairwise_sq_dists / self.sigma_sq)

    def fit_transform(self, x, *_):
        return self.fit(x).transform(x)


# kernel function
def kernel(x, kernel_type='poly', basis_func_param=2, random_state=123):
    if 'poly' == kernel_type:
        poly = PolynomialFeatures(basis_func_param, include_bias=False)
        poly.fit(x)
        return poly, poly.transform(x)
    elif 'rbf' == kernel_type:
        rbf = RbfFeatures(basis_func_param, random_state)
        rbf.fit(x)
        return rbf, rbf.transform(x)
    return None, x


# the gradient of the kernel function
def kernel_gradient(transformer, type='poly'):
    if 'poly' == type:
        transformed_function, input_feature_names = get_polynomial_feature_names(transformer)

        input_features = []
        for input_feature in input_feature_names:
            locals()['{}'.format(input_feature)] = sy.symbols('{}'.format(input_feature))
            input_features.append(locals()['{}'.format(input_feature)])

        func = sy.Matrix([])
        for name in transformed_function:
            func = func.row_join(sy.Matrix([[eval(name)]]))

        func_grad = func.jacobian(input_features)
        f = sy.lambdify([input_features], func_grad, 'numpy')
        return f
    elif 'rbf' == type:
        return transformer


def rbf_gradient(train_x_bar, phi_x_bar, kernel):
    temp = np.subtract(train_x_bar[:, np.newaxis, :], kernel.centroids[np.newaxis, :, :])
    return -(2 / kernel.sigma_sq) * np.multiply(temp, phi_x_bar[:, :, np.newaxis])


def learn_model(x, y, reg, beta=None):
    reg_learner = Ridge(alpha=reg, fit_intercept=True)
    reg_learner.fit(x, y, sample_weight=beta)
    return reg_learner


# get the transformed feature names
def get_polynomial_feature_names(poly):
    """
    Return feature names for output features

    Parameters
    ----------
    poly : the polynomial transformer

    Returns
    -------
    output_feature_names : list of string, length n_output_features

    """
    powers = poly.powers_
    input_feature_names = ['x%d' % i for i in range(powers.shape[1])]
    feature_names = []
    for row in powers:
        inds = np.where(row)[0]
        if len(inds):
            name = "*".join("%s**%d" % (input_feature_names[ind], exp)
                            if exp != 1 else input_feature_names[ind]
                            for ind, exp in zip(inds, row[inds]))
        else:
            name = "1"
        feature_names.append(name)
    return feature_names, input_feature_names


# def compute_grad_j_w_w(phi_x_tilde, reg_constant):
#     temp = np.matmul(phi_x_tilde[:, :, np.newaxis], phi_x_tilde[:, np.newaxis, :])
#     temp = 2 * np.mean(temp, axis=0)
#
#     # alternative calculation
#     # def gradient_per_sample(x):
#     #     return x.transpose() @ x
#     #
#     # w_size = phi_x_tilde.shape[1]
#     # temp = np.zeros((w_size, w_size))
#     #
#     # for idx, x in enumerate(phi_x_tilde):
#     #     temp += gradient_per_sample(x.reshape(1, -1))
#     # temp = 2 * temp / phi_x_tilde.shape[0]
#     grad_j_w_w = temp + reg_constant * np.eye(phi_x_tilde.shape[1])
#     return grad_j_w_w
#
#
# def compute_grad_j_w_b(phi_x_tilde):
#     return 2 * np.mean(phi_x_tilde, axis=0)


def compute_grad_j_w_w(phi_x_bar, reg_constant, grad_j_w_w_clean, n):
    temp = np.matmul(phi_x_bar[:, :, np.newaxis], phi_x_bar[:, np.newaxis, :])
    temp = (2 * np.sum(temp, axis=0) / n) + grad_j_w_w_clean
    grad_j_w_w = temp + reg_constant * np.eye(phi_x_bar.shape[1])
    return grad_j_w_w


def compute_grad_j_w_b(phi_x_bar, grad_j_w_b_x_clean, n):
    return (2 * np.sum(phi_x_bar, axis=0) / n) + grad_j_w_b_x_clean


def compute_grad_j_theta_x(clf, phi_x_bar, train_x_bar, train_y_bar, kernel_grad_function, basis_function):
    if 'poly' == basis_function:
        grad_phi_x_x = np.apply_along_axis(kernel_grad_function, axis=1, arr=train_x_bar)
    elif 'rbf' == basis_function:
        grad_phi_x_x = rbf_gradient(train_x_bar, phi_x_bar, kernel_grad_function)
    w_t, b = clf.coef_, clf.intercept_

    w_t = np.tile(w_t, (phi_x_bar.shape[0], 1))
    temp = 2 * np.matmul(w_t[:, np.newaxis, :], phi_x_bar[:, :, np.newaxis]).reshape(-1, 1) + b - train_y_bar
    grad_j_w_x = temp[:, np.newaxis] * grad_phi_x_x
    grad_j_b_x = np.matmul(w_t[:, np.newaxis, :], grad_phi_x_x)

    # alternative calculation
    # phi_x_bar = np.reshape(phi_x_bar, phi_x_bar.shape + (1,))
    # grad_phi_x_x = np.apply_along_axis(kernel_grad_function, axis=1, arr=train_x_bar)
    # w_t, b = clf.coef_, clf.intercept_
    #
    # def calc_grad_j_w_x(phi_x_i, grad_phi_x_x_i, i):
    #     return (2 * w_t @ phi_x_i + b - train_y_bar[i]) * grad_phi_x_x_i
    #
    # grad_j_w_x = np.array(
    #     [calc_grad_j_w_x(phi_x_bar[i, :, :], grad_phi_x_x[i, :], i) for i in range(phi_x_bar.shape[0])])
    #
    # def calc_grad_j_b_x(grad_phi_x_x_i):
    #     return w_t @ grad_phi_x_x_i
    #
    # grad_j_b_x = np.array([calc_grad_j_b_x(grad_phi_x_x_i) for grad_phi_x_x_i in grad_phi_x_x])

    return grad_j_w_x, grad_j_b_x


def compute_grad_theta_z_bar(hessian_mat, n, grad_j_w_x, grad_j_b_x, phi_x_bar):
    grad_j_w_y = -np.reshape(phi_x_bar, phi_x_bar.shape + (1,))
    grad_j_b_y = np.ones((grad_j_b_x.shape[0], 1, 1)) * -1

    row_1 = np.concatenate((grad_j_w_x, grad_j_w_y), axis=2)
    row_2 = np.concatenate((grad_j_b_x, grad_j_b_y), axis=2)

    grad_j_theta_z = np.concatenate((row_1, row_2), axis=1)
    grad_j_theta_z = -(2 / n) * grad_j_theta_z

    def grad_theta_xy(rhs):
        return np.linalg.lstsq(hessian_mat, rhs, rcond=None)[0]

    grad_theta_z_bar = np.array([grad_theta_xy(grad_j_theta_z_i) for grad_j_theta_z_i in grad_j_theta_z])

    # alternative calculation
    # hessian_inv = np.linalg.inv(hessian_mat)
    # grad_theta_z_bar1 = np.matmul(hessian_inv, grad_j_theta_z)

    # grad_w_x_bar = grad_theta_z_bar[:, :-1, :-1]  # get all but last row and column
    # grad_w_y_bar = grad_theta_z_bar[:, :-1, -1]  # get last column, excluding the last row
    # grad_b_x_bar = grad_theta_z_bar[:, -1, :-1]  # get last row, excluding the last column
    # grad_b_y_bar = grad_theta_z_bar[:, -1, -1]
    #
    # return grad_theta_z_bar, grad_w_x_bar, grad_b_x_bar, grad_w_y_bar, grad_b_y_bar

    return grad_theta_z_bar


def comp_objective_e(train_x_bar, train_x, y_tilde, phi_x_valid, valid_y, transformer, reg_constant):
    x_tilde = np.vstack((train_x, train_x_bar))
    phi_x_tilde = transformer.transform(x_tilde)
    clf = learn_model(phi_x_tilde, y_tilde, reg_constant)
    valid_y_pred = clf.predict(phi_x_valid)
    return mean_squared_error(valid_y, valid_y_pred)


def compute_attack_gradient(clf, grad_theta_z_bar, phi_valid_x, valid_y):
    residual = 2 * (clf.predict(phi_valid_x).reshape(-1, 1) - valid_y)

    grad_e_w = np.mean(np.multiply(residual, phi_valid_x), axis=0)
    grad_e_b = np.mean(residual)
    grad_e_theta = np.concatenate((grad_e_w.reshape(-1, 1), grad_e_b.reshape(-1, 1)), axis=0)

    grad_e_theta = np.tile(grad_e_theta.transpose(), (grad_theta_z_bar.shape[0], 1))
    grad_e_z = np.matmul(np.transpose(grad_theta_z_bar, (0, 2, 1)), grad_e_theta[:, :, np.newaxis])

    return np.squeeze(grad_e_z)
