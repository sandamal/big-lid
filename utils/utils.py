import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split

mpl.use('Agg')


def sample_dataset(data, train_set_size, test_set_size, validation_set_size, poison_per, random_seed):
    poison_count = int(0.5 + train_set_size * (poison_per / (1 - poison_per)))

    data = data[:train_set_size + poison_count + test_set_size + validation_set_size, :]

    train_X, remaining_X, train_y, remaining_y = train_test_split(data[:, :-1], data[:, -1],
                                                                  train_size=train_set_size + poison_count,
                                                                  test_size=test_set_size + validation_set_size,
                                                                  random_state=random_seed)

    valid_X, test_X, valid_y, test_y = train_test_split(remaining_X, remaining_y,
                                                        train_size=validation_set_size,
                                                        test_size=test_set_size,
                                                        random_state=random_seed)

    return train_X, train_y, test_X, test_y, valid_X, valid_y, poison_count


def sample_initial_poison_points(train_x, train_y, clean_train_set_size, poison_count, random_seed):
    train_X, train_X_bar, train_y, train_y_bar = train_test_split(train_x, train_y,
                                                                  train_size=clean_train_set_size,
                                                                  test_size=poison_count,
                                                                  random_state=random_seed)

    return train_X, train_y, train_X_bar, train_y_bar


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
