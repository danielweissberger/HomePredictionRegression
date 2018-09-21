import scipy.stats as stats
import numpy as np

def get_data_to_compare(data_list):
    data_lists_for_eval = []
    for i in range(0, len(data_list)):
        for j in range(0, len(data_list)):
            if i != j:
                data_lists_for_eval.append([data_list[i], data_list[j]])

    for var in data_lists_for_eval:
        var = var.sort()

    data_sets_for_eval = set(tuple(x) for x in data_lists_for_eval)
    data_lists_for_eval = [list(x) for x in data_sets_for_eval]

    return list(data_lists_for_eval)


# This function will return the p-value and hypothesis result (true/false for alternate)
# for a signifigance test comparing two sample means, given null hypothesis that the means should be equal

def calculate_p_for_mean_diff(v1, v2, alpha, df, target_var, target_col):
    data1 = df[df[target_col] == v1][target_var]
    data2 = df[df[target_col] == v2][target_var]

    data1_mean = np.mean(data1)
    data2_mean = np.mean(data2)
    data1_std = np.std(data1)
    data2_std = np.std(data2)
    diff_of_means = data1_mean - data2_mean
    sigma = np.sqrt((data1_std ** 2) / len(data1) + (data2_std ** 2) / len(data2))
    z = (diff_of_means - 0) / sigma

    if z < 0:
        p_val_from_z = stats.norm.cdf(z) * 2
    else:
        p_val_from_z = (1 - stats.norm.cdf(z)) * 2

    return v1, v2, p_val_from_z, p_val_from_z < alpha


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr
    corr = stats.pearsonr(x,y)

    # Return the pearson value
    return corr[0]


def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))


def bootstrap_replicate_2d(data, func):
    # Create an array for index selection
    idx_array = np.array(range(0, len(data[0])))

    # randomize an array of indices for selection of x,y pairs
    random_indices = np.random.choice(idx_array, size=len(idx_array))

    # Select x,y paira and call 2d function
    d1 = data[0].take(random_indices)
    d2 = data[1].take(random_indices)
    return func(d1, d2)


def draw_bs_reps(data, func, bootstrap_generator, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_generator(data, func)

    return bs_replicates