'''
    Description: Compare the defenses
'''
import os

import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.metrics import mean_squared_error

from defenses import ransac_model, huber_model, theilsen_model, trimclf
from utils.attack_utilities import compute_grad_j_w_w, \
    compute_grad_j_w_b, compute_grad_theta_z_bar, compute_attack_gradient, kernel_gradient, \
    compute_grad_j_theta_x
from utils.lid_utils import get_lids
from utils.utils import unison_shuffled_copies
from utils.attack_utilities import kernel, learn_model

defenses = ['no_defense', 'big-lid', 'TRIM', 'RANSACRegressor', 'HuberRegressor']
# the weights used by big-lid
weight_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Dataframe to hold results
column_names = ['scenario', 'cv_set', 'poison_perc', 'tolerence', 'MSE', 'MSE_inc']
df = pd.DataFrame(columns=column_names)

# parameters of dataset
dataset = 'heart_disease'
basis_function = 'rbf'

# attack parameters
poison_percentages = [0.02, 0.04, 0.06, 0.08, 0.1]
attack_tolerence = 0.1

random_seeds = [1234, 1969, 1989, 2000, 4]

clean_train_set_size = None
test_set_size = None
validation_set_size = None
reg_constant = None
basis_function_param = None

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

for cv, random_seed in enumerate(random_seeds):
    cv += 1
    for idx, poison_per in enumerate(poison_percentages):
        # Dataframe to hold iteration results
        iter_column_names = ['iteration', 'benign_count', 'benign \%', 'poison_count', 'poison \%',
                             'poisoned_removed_mse', 'Poison removed', 'benign_removed_mse',
                             'Benign removed', 'total_mse', 'Overall effect', 'test_mse']
        df_iteration = pd.DataFrame(columns=iter_column_names)

        # data file
        file_name = '{}_{}_{}_{}'.format(dataset, basis_function, poison_per, attack_tolerence)
        data = np.load(f'./datasets/{dataset}/CV_{cv}/{poison_per}/{file_name}.npz')

        # load the attacked data
        z_bar_initial = data['z_bar_initial']
        z_bar = data['z_bar']
        train_x = data['train_x']
        train_y = data['train_y']
        valid_x = data['valid_x']
        valid_y = data['valid_y']
        test_x = data['test_x']
        test_y = data['test_y']

        clean_train_set_size = train_x.shape[0]
        test_set_size = test_x.shape[0]
        validation_set_size = valid_x.shape[0]
        poisoned_size = z_bar.shape[0]

        x_bar = z_bar[:, :-1]
        y_bar = z_bar[:, -1].reshape(-1, 1)
        x_tilde = np.vstack((train_x, x_bar))
        y_tilde = np.vstack((train_y, y_bar))
        z_tilde = np.hstack((x_tilde, y_tilde))

        transformer, phi_x = kernel(train_x, basis_function, basis_function_param, random_seed)
        phi_x_bar = transformer.transform(x_bar)
        phi_x_valid = transformer.transform(valid_x)
        phi_x_test = transformer.transform(test_x)
        phi_x_tilde = np.vstack((phi_x, phi_x_bar))
        phi_z_tilde = np.hstack((phi_x_tilde, y_tilde))

        # LID GMM defense
        # calculate the gradient of validation error w.r.t z_tilde
        # assume that we are poisoning all the samples
        kernel_grad_function = kernel_gradient(transformer, basis_function)
        n = x_tilde.shape[0]
        grad_j_w_w_x_clean = 0
        grad_j_w_b_x_clean = 0

        # Calculate grad^2 J w.r.t theta (Equation (10)), this does not change with the model
        grad_j_w_w = compute_grad_j_w_w(phi_x_tilde, reg_constant, grad_j_w_w_x_clean, n)
        grad_j_w_b = compute_grad_j_w_b(phi_x_tilde, grad_j_w_b_x_clean, n)

        grad_j_w_b = grad_j_w_b.reshape((-1, 1))
        grad_j_b_w = grad_j_w_b.transpose()

        row_1 = np.concatenate((grad_j_w_w, grad_j_w_b), axis=1)
        row_2 = np.concatenate((grad_j_b_w, np.array([2]).reshape(1, 1)), axis=1)
        hessian_mat = np.concatenate((row_1, row_2), axis=0)

        z_tilde_shape = z_tilde.shape
        previous_model = best_model = learn_model(phi_x_tilde, y_tilde, reg_constant)
        valid_y_pred = previous_model.predict(phi_x_valid)
        previous_mse = validation_mse = mean_squared_error(valid_y, valid_y_pred.ravel())

        try:
            data = np.load(f'lid_output/{file_name}.npz')
            input_space_lids_5 = data['input_space_lids_5']
            input_space_lids_10 = data['input_space_lids_10']
            input_space_lids_15 = data['input_space_lids_15']
            input_space_lids_20 = data['input_space_lids_20']
            input_space_lids_25 = data['input_space_lids_25']
        except:
            print('calculating lids')
            input_space_lids_5 = get_lids(x_tilde, 5, None)
            input_space_lids_10 = get_lids(x_tilde, 10, None)
            input_space_lids_15 = get_lids(x_tilde, 15, None)
            input_space_lids_20 = get_lids(x_tilde, 20, None)
            input_space_lids_25 = get_lids(x_tilde, 25, None)
            np.savez('lid_output/{}.npz'.format(file_name), input_space_lids_5=input_space_lids_5,
                     input_space_lids_10=input_space_lids_10, input_space_lids_15=input_space_lids_15,
                     input_space_lids_20=input_space_lids_20, input_space_lids_25=input_space_lids_25)

        lid_values = [input_space_lids_5, input_space_lids_10, input_space_lids_15, input_space_lids_20,
                      input_space_lids_25]


        # function to calculate the derivative of validation error w.r.t Z
        def calculate_derivative(pois_clf):

            grad_j_w_x, grad_j_b_x = compute_grad_j_theta_x(pois_clf, phi_x_tilde, x_tilde, y_tilde,
                                                            kernel_grad_function, basis_function)
            # compute partial derivatives
            grad_theta_z_bar = compute_grad_theta_z_bar(hessian_mat, n, grad_j_w_x, grad_j_b_x, phi_x_tilde)
            gradient_e_z = compute_attack_gradient(pois_clf, grad_theta_z_bar, phi_x_valid, valid_y)
            return -gradient_e_z


        iteration = 0
        top_count = 200
        suspected_indices = []
        iteration_count = 100
        best_test_mse = 10

        while iteration < iteration_count:
            grad_e_z = calculate_derivative(previous_model)
            grad_e_z[suspected_indices, :] = 0

            # get the gradient w.r.t x
            grad_norms = np.linalg.norm(grad_e_z[:, :-1], axis=1)
            # get the gradient w.r.t z
            grad_norms = np.hstack((grad_norms.reshape(-1, 1), grad_e_z[:, -1].reshape(-1, 1)))
            grad_norms = np.linalg.norm(grad_norms, axis=1)

            # sort the gradients in descending order
            sorted_indx = np.argsort(grad_norms)[::-1]
            # the samples with the largest norms are at the end
            high_sensitive_indices = sorted_indx[:top_count]
            # the samples with the smallest norms
            low_sensitive_indices = sorted_indx[top_count:]

            selected_indices_to_weight = None
            selected_model_for_lid = None
            selected_model_for_lid_mse = 10
            selected_weights_for_iteration = None

            # Pick the LID k
            for calculated_lids in lid_values:
                largest_lids = calculated_lids[high_sensitive_indices]
                log_lids = np.log(largest_lids).reshape(-1, 1)
                gmm = mixture.GaussianMixture(n_components=2, n_init=10).fit(log_lids)
                responsibilities = gmm.predict_proba(log_lids)

                responsibilities_of_max = gmm.predict_proba(np.max(log_lids).reshape(1, -1)).ravel()
                weights = responsibilities[:, np.argmax(responsibilities_of_max, axis=0)]
                threshold = 0.98
                indices_to_down_weight_in_iteration = np.argwhere(weights >= threshold)
                while len(indices_to_down_weight_in_iteration) == 0:
                    threshold -= 0.02
                    indices_to_down_weight_in_iteration = np.argwhere(weights >= threshold)
                indices_to_down_weight_in_iteration = high_sensitive_indices[indices_to_down_weight_in_iteration]
                suspected_indices_copy = suspected_indices.copy()
                suspected_indices_copy.extend(indices_to_down_weight_in_iteration.ravel())

                # Pick the weight
                current_model_for_lid = current_weights_for_lid = None
                current_mse_for_lid = 10
                selected_weight_model = None
                for weight in weight_values:
                    weights = np.ones_like(y_tilde)
                    weights[suspected_indices_copy] = weight
                    current_model_for_weight = learn_model(phi_x_tilde, y_tilde, reg_constant, beta=weights.ravel())
                    valid_y_pred = current_model_for_weight.predict(phi_x_valid)
                    current_mse_for_weight = mean_squared_error(valid_y, valid_y_pred.ravel())
                    if current_mse_for_weight < current_mse_for_lid:
                        current_model_for_lid = current_model_for_weight
                        current_mse_for_lid = current_mse_for_weight
                        current_weights_for_lid = weights

                # Check if this is the best MSE for the given LID values
                if current_mse_for_lid < selected_model_for_lid_mse:
                    selected_model_for_lid_mse = current_mse_for_lid
                    selected_model_for_lid = current_model_for_lid
                    selected_indices_to_weight = indices_to_down_weight_in_iteration
                    selected_weights_for_iteration = current_weights_for_lid

            indices_to_down_weight_in_iteration = selected_indices_to_weight
            current_mse_for_iteration = selected_model_for_lid_mse
            current_model_for_iteration = selected_model_for_lid
            suspected_indices.extend(indices_to_down_weight_in_iteration.ravel())

            # Select the correct iteration to stop
            if current_mse_for_iteration <= previous_mse:
                previous_mse = current_mse_for_iteration
                best_model = current_model_for_iteration
                selected_weights = selected_weights_for_iteration
            previous_model = current_model_for_iteration

            iteration += 1

        ###############################################################
        # No Attack
        ###############################################################
        print('running no attack')
        clf = learn_model(phi_x, train_y, reg_constant)
        test_y_pred = clf.predict(phi_x_test)
        mse_initial = mean_squared_error(test_y, test_y_pred.ravel())
        df = df.append(
            pd.DataFrame([['no_attack', cv, poison_per, attack_tolerence, mse_initial, 0]], columns=column_names))
        print(f'no attack test error - {mse_initial}')

        ###############################################################
        # No Defense
        ###############################################################
        if 'no_defense' in defenses:
            print('running no defense')
            clf = learn_model(phi_x_tilde, y_tilde, reg_constant)
            test_y_pred = clf.predict(phi_x_test)
            mse = mean_squared_error(test_y, test_y_pred.ravel())
            mse_increase = 100 * (mse - mse_initial) / mse_initial
            df = df.append(pd.DataFrame([[f'no_defense', cv, poison_per, attack_tolerence, mse, mse_increase]],
                                        columns=column_names))

        ###############################################################
        # LID DEFENSE
        ###############################################################
        if 'big-lid' in defenses:
            clf = best_model
            test_y_pred = clf.predict(phi_x_test)
            mse = mean_squared_error(test_y, test_y_pred.ravel())
            mse_increase = 100 * (mse - mse_initial) / mse_initial
            df = df.append(
                pd.DataFrame([[f'big-lid', cv, poison_per, attack_tolerence, mse, mse_increase]],
                             columns=column_names))

        ###############################################################
        # RANSAC
        ###############################################################
        if 'RANSACRegressor' in defenses:
            print('running RANSAC')
            X_copy, y_copy = unison_shuffled_copies(np.copy(phi_x_tilde), np.copy(y_tilde))
            count_values = [0.2, 0.4, 0.6, 0.8]
            ridge_model = learn_model(phi_x_tilde, y_tilde, reg_constant)
            reg_learner, elapsed_time = ransac_model(X_copy, y_copy, ridge_model, count_values)
            test_y_pred = reg_learner.predict(phi_x_test)
            mse = mean_squared_error(test_y, test_y_pred.ravel())
            mse_increase = 100 * (mse - mse_initial) / mse_initial
            df = df.append(
                pd.DataFrame([[f'RANSACRegressor', cv, poison_per, attack_tolerence, mse, mse_increase]],
                             columns=column_names))

        ###############################################################
        # Huber
        ###############################################################
        if 'HuberRegressor' in defenses:
            print('running HUBER')
            X_copy, y_copy = unison_shuffled_copies(np.copy(phi_x_tilde), np.copy(y_tilde))
            epsilon_values = [1.35, 1.5, 1.75, 1.9]
            reg_learner, elapsed_time = huber_model(X_copy, y_copy, epsilon_values, reg_constant)
            test_y_pred = reg_learner.predict(phi_x_test)
            mse = mean_squared_error(test_y, test_y_pred.ravel())
            mse_increase = 100 * (mse - mse_initial) / mse_initial
            df = df.append(
                pd.DataFrame([[f'HuberRegressor', cv, poison_per, attack_tolerence, mse, mse_increase]],
                             columns=column_names))

        ###############################################################
        # TheilSenRegressor
        ###############################################################
        if 'TheilSenRegressor' in defenses:
            print('running TheilSenRegressor')
            X_copy, y_copy = unison_shuffled_copies(np.copy(phi_x_tilde), np.copy(y_tilde))
            num_feats = X_copy.shape[1]
            n_subsamples = [num_feats, num_feats + 100, num_feats + 200, num_feats + 300, num_feats + 400]
            reg_learner, elapsed_time = theilsen_model(X_copy, y_copy, n_subsamples)
            test_y_pred = reg_learner.predict(phi_x_test)
            mse = mean_squared_error(test_y, test_y_pred.ravel())
            mse_increase = 100 * (mse - mse_initial) / mse_initial
            df = df.append(
                pd.DataFrame([[f'TheilSenRegressor', cv, poison_per, attack_tolerence, mse, mse_increase]],
                             columns=column_names))

        ###############################################################
        # TRIM
        ###############################################################
        if 'TRIM' in defenses:
            print('running TRIM')
            trim_clf, lam, ind = trimclf(phi_x_tilde, y_tilde, clean_train_set_size, reg_constant)
            test_y_pred = trim_clf.predict(phi_x_test)
            mse = mean_squared_error(test_y, test_y_pred.ravel())
            mse_increase = 100 * (mse - mse_initial) / mse_initial
            df = df.append(
                pd.DataFrame([[f'TRIM', cv, poison_per, attack_tolerence, mse, mse_increase]],
                             columns=column_names))

directory = f'./results/'
os.makedirs(directory, exist_ok=True)
full_results = f'./results/res_{dataset}_tol_{attack_tolerence}.csv'
with open(full_results, 'w') as f:
    df = df.sort_values('scenario')
    df.to_csv(f, encoding='utf-8', index=False)

# Dataframe to hold results
column_names = ['scenario', 'poison_percentage', 'avg_MSE', 'std_MSE', 'avg_MSE_increase', 'std_MSE_increase']
df_results = pd.DataFrame(columns=column_names)

for defense in defenses:
    learner_df = df[df.scenario == defense]
    for poison_percentage in poison_percentages:
        temp_df = learner_df[learner_df.poison_perc == poison_percentage]
        mean_mse = temp_df['MSE'].mean()
        std_mse = temp_df['MSE'].std()
        mean_mse_increase = temp_df['MSE_inc'].mean()
        std_mse_increase = temp_df['MSE_inc'].std()
        df_results = df_results.append(
            pd.DataFrame([[defense, poison_percentage, mean_mse, std_mse, mean_mse_increase, std_mse_increase]],
                         columns=column_names))

summary_results = f'./results/res_{dataset}_tol_{attack_tolerence}_summarized.csv'

with open(summary_results, 'w+') as f:
    df_results.to_csv(f, encoding='utf-8', index=False)
