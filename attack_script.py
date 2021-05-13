'''
    Description: This is the main attack script
'''
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import line_search
from sklearn.metrics import mean_squared_error

from utils.attack_utilities import kernel, compute_grad_j_w_w, \
    compute_grad_j_w_b, learn_model, compute_grad_theta_z_bar, compute_attack_gradient, kernel_gradient, \
    compute_grad_j_theta_x
from utils.utils import sample_dataset, sample_initial_poison_points

# gradient descent implementation or scipy optimize
opt_method = 'gd'
adaptive = True
print_mse = False

# parameters of dataset
dataset = 'heart_disease'
basis_function = 'rbf'
random_seeds = [1234, 69, 1989, 2000, 4]

if 'appliances' == dataset:
    clean_train_set_size = 4000
    test_set_size = 1500
    validation_set_size = 1500
    reg_constant = 0.0010608183551394483
    basis_function_param = 60
    if 'poly' == basis_function:
        reg_constant = 10
        basis_function_param = 3
elif 'house' == dataset:
    clean_train_set_size = 1400
    test_set_size = 700
    validation_set_size = 700
    reg_constant = 0.0004375479375074184
    basis_function_param = 310
    if 'poly' == basis_function:
        reg_constant = 10
        basis_function_param = 2
elif 'heart_disease' == dataset:
    clean_train_set_size = 1400
    test_set_size = 700
    validation_set_size = 700
    reg_constant = 0.00018047217668271703
    basis_function_param = 120
    if 'poly' == basis_function:
        reg_constant = 0.2894266124716749
        basis_function_param = 4

# attack parameters
poison_percentages = [0.02, 0.04, 0.06, 0.08, 0.1]
tol_values = [0.1]

data = np.load('./datasets/{}.npy'.format(dataset))


def poison_data():
    for poison_per in poison_percentages:
        # split the dataset
        train_x, train_y, test_x, test_y, valid_x, valid_y, poison_count = sample_dataset(data,
                                                                                          clean_train_set_size,
                                                                                          test_set_size,
                                                                                          validation_set_size,
                                                                                          poison_per,
                                                                                          random_seed)

        # random sample initial poison points
        train_x, train_y, train_x_bar_0, train_y_bar_0 = sample_initial_poison_points(train_x, train_y,
                                                                                      clean_train_set_size,
                                                                                      poison_count, random_seed)

        # reshape the label vectors
        train_y = train_y.reshape(-1, 1)
        train_y_bar_0 = train_y_bar_0.reshape(-1, 1)
        test_y = test_y.reshape(-1, 1)
        valid_y = valid_y.reshape(-1, 1)

        transformer, phi_x = kernel(train_x, basis_function, basis_function_param, random_seed)
        phi_x_test = transformer.transform(test_x)
        phi_x_valid = transformer.transform(valid_x)

        kernel_grad_function = kernel_gradient(transformer, basis_function)

        if print_mse:
            # Learn the model using data prior to poisoning
            clf = learn_model(phi_x, train_y, reg_constant)
            test_y_pred = clf.predict(phi_x_test)
            mse_test_initial = mean_squared_error(test_y, test_y_pred)

        # precalculate to reduce computation time
        x_tilde = np.vstack((train_x, train_x_bar_0))
        n = x_tilde.shape[0]
        grad_j_w_w_x_clean = np.matmul(phi_x[:, :, np.newaxis], phi_x[:, np.newaxis, :])
        grad_j_w_w_x_clean = 2 * np.sum(grad_j_w_w_x_clean, axis=0) / n
        grad_j_w_b_x_clean = 2 * np.sum(phi_x, axis=0) / n

        # posed as a minimization problem
        def validation_error(z_bar_update):
            z_bar_update = z_bar_update.reshape(z_bar_shape)
            x_bar_update = z_bar_update[:, :-1]
            y_bar_update = z_bar_update[:, -1].reshape(-1, 1)

            phi_x_bar_update = transformer.transform(x_bar_update)
            phi_x_tilde_update = np.vstack((phi_x, phi_x_bar_update))
            y_tilde_update = np.vstack((train_y, y_bar_update))

            pois_clf = learn_model(phi_x_tilde_update, y_tilde_update, reg_constant)
            valid_y_predictions = pois_clf.predict(phi_x_valid)
            mse = mean_squared_error(valid_y, valid_y_predictions)
            return -mse

        # posed as a minimization problem
        def calculate_derivative(z_bar_update):
            z_bar_update = z_bar_update.reshape(z_bar_shape)
            x_bar_update = z_bar_update[:, :-1]
            y_bar_update = z_bar_update[:, -1].reshape(-1, 1)

            phi_x_bar_update = transformer.transform(x_bar_update)
            phi_x_tilde_update = np.vstack((phi_x, phi_x_bar_update))
            y_tilde_update = np.vstack((train_y, y_bar_update))

            # Calculate grad^2 J w.r.t theta (Equation (10))
            grad_j_w_w = compute_grad_j_w_w(phi_x_bar_update, reg_constant, grad_j_w_w_x_clean, n)
            grad_j_w_b = compute_grad_j_w_b(phi_x_bar_update, grad_j_w_b_x_clean, n)

            grad_j_w_b = grad_j_w_b.reshape((-1, 1))
            grad_j_b_w = grad_j_w_b.transpose()

            row_1 = np.concatenate((grad_j_w_w, grad_j_w_b), axis=1)
            row_2 = np.concatenate((grad_j_b_w, np.array([2]).reshape(1, 1)), axis=1)
            hessian_mat = np.concatenate((row_1, row_2), axis=0)

            pois_clf = learn_model(phi_x_tilde_update, y_tilde_update, reg_constant)

            grad_j_w_x, grad_j_b_x = compute_grad_j_theta_x(pois_clf, phi_x_bar_update, x_bar_update,
                                                            y_bar_update,
                                                            kernel_grad_function, basis_function)
            # compute partial derivatives
            grad_theta_z_bar = compute_grad_theta_z_bar(hessian_mat, n, grad_j_w_x, grad_j_b_x, phi_x_bar_update)

            grad_e_z = compute_attack_gradient(pois_clf, grad_theta_z_bar, phi_x_valid, valid_y)
            return -grad_e_z.flatten()

        for tol in tol_values:
            # data file
            file_name = '{}_{}_{}_{}'.format(dataset, basis_function, poison_per, tol)

            z_bar_initial = np.array(np.concatenate((train_x_bar_0, train_y_bar_0), axis=1))
            z_bar_shape = z_bar_initial.shape
            z_bar_starting = z_bar_initial.flatten()

            phi_x_bar_initial = transformer.transform(train_x_bar_0)
            phi_x_tilde_initial = np.vstack((phi_x, phi_x_bar_initial))
            y_tilde_initial = np.vstack((train_y, train_y_bar_0))

            poisoned_clf = learn_model(phi_x_tilde_initial, y_tilde_initial, reg_constant)
            valid_y_pred = poisoned_clf.predict(phi_x_valid)
            mse_initial = -mean_squared_error(valid_y, valid_y_pred)
            step = 0.5  # step size multiplier
            precision = 1e-6  # desired precision of result
            max_iters = 1000  # maximum number of iterations

            # initializing variables
            last_z = next_z = z_bar_starting
            last_f = mse_initial
            next_f = None

            all_f_i = list()
            all_f_i.append(mse_initial)

            # gradient descent fixed step size
            for i in range(max_iters):
                print('iteration {}'.format(i))
                current_z = next_z
                dz_i = calculate_derivative(current_z)

                if adaptive:
                    step = line_search(validation_error, calculate_derivative, current_z, -dz_i, dz_i)
                    next_f = step[3]
                    step = step[0]
                    if step is None:
                        step = 0

                next_z = current_z - step * dz_i
                # maintain attack severity constraint
                diff_z = next_z - z_bar_starting
                signs = np.sign(diff_z)
                threshold_violations = np.where(np.abs(diff_z) > tol)[0]
                next_z[threshold_violations] = z_bar_starting[threshold_violations] + signs[
                    threshold_violations] * tol
                next_z = np.clip(next_z, 0, 1)

                if next_f is None or not adaptive:
                    next_f = validation_error(next_z)
                if next_f - last_f > 0:
                    next_z = last_z
                    break
                if abs(next_f - last_f) <= precision:
                    break
                all_f_i.append(next_f)

                last_z = next_z
                last_f = next_f

            optimized_z = np.clip(next_z, 0, 1)
            optimized_z = optimized_z.reshape(z_bar_shape)

            directory = f'datasets/{dataset}/CV_{CV}/{poison_per}/'
            os.makedirs(directory, exist_ok=True)
            np.savez(f'datasets/{dataset}/CV_{CV}/{poison_per}/{file_name}', z_bar_initial=z_bar_initial,
                     z_bar=optimized_z,
                     train_x=train_x,
                     train_y=train_y, valid_x=valid_x, valid_y=valid_y, test_x=test_x, test_y=test_y)

            if print_mse:
                # check if z_bar has changed from the attack
                print(np.array_equal(optimized_z, z_bar_initial))
                print(np.allclose(optimized_z, z_bar_initial))

                # Learn the model using data after poisoning
                x_tilde = np.vstack((train_x, optimized_z[:, :-1]))
                y_tilde = np.vstack((train_y, optimized_z[:, -1].reshape(-1, 1)))
                phi_x_tilde = transformer.transform(x_tilde)
                clf = learn_model(phi_x_tilde, y_tilde, reg_constant)
                test_y_pred = clf.predict(phi_x_test)
                mse_test_after = mean_squared_error(test_y, test_y_pred)

                print(f'{poison_per}_{tol} - percentage and tolerance')
                print(f'{mse_test_initial} - initial MSE')
                print(f'{mse_test_after} - MSE')
                print(f'{mse_test_after - mse_test_initial} - Difference')
        plt.close()


for j, seed in enumerate(random_seeds):
    CV = j + 1
    random_seed = seed
    poison_data()
