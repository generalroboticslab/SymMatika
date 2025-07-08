
#include "differentiation.h"

std::unordered_map<int, std::string> variable_pairing_map;
size_t pd_matrix_columns;

// construct variable pairing map
void initialize_pairing_map(std::unordered_map<int, std::string> &variable_pairing_map, const DataSet &dataset) {
    int counter = 0;
    for (const auto& var_i : dataset.var_list) {
        for (const auto& var_j : dataset.var_list) {
            if (var_i != var_j) {
                variable_pairing_map[counter] = var_i + "/" + var_j;
                counter++;
            }
        }
    }
}

// numerically approximates partial derivative pairs
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> num_pds(const DataSet &dataset) {
    size_t n_vars = dataset.var_list.size();

    size_t n_rows = n_vars * (n_vars - 1) / 2;
    size_t n_columns = dataset.pd_matrix_columns;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> numeric_partials(n_rows, n_columns);

    size_t row_index = 0;
    for (size_t i=0; i<n_vars; i++) {
        for (size_t j=i+1; j<n_vars; j++) {
            if (i != j) [[likely]] {
                size_t col_index = 0;
                for (const auto& trial_matrix : dataset.data) {
                    size_t n_points = trial_matrix.cols();
                    for (size_t p=1; p<n_points-1; p++) {
                        double di = trial_matrix(i, p+1) - trial_matrix(i, p-1);
                        double dj = trial_matrix(j, p+1) - trial_matrix(j, p-1);

                        if (std::abs(dj)>1e-10) [[likely]] {
                            numeric_partials(row_index, col_index) = di/dj;
                        } else {
                            numeric_partials(row_index, col_index) = std::numeric_limits<double>::quiet_NaN();
                        }
                        col_index++;
                    }
                }
                row_index++;
            }
        }
    }
    return numeric_partials;
}

// for low-dimensional systems (i.e. 2 variables)
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sym_pds(const DataSet &dataset, std::unordered_map<int, RCP<const Basic>> &gradient) {
    size_t n_vars = dataset.var_list.size();
    size_t n_rows = n_vars * (n_vars - 1) / 2;
    size_t n_columns = dataset.pd_matrix_columns;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pd_matrix(n_rows, n_columns);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> copy_pd_matrix = pd_matrix;

    UnsupervisedEvaluator evaluator;
    evaluator.prepare_gradient(gradient);

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gradients;
    gradients.resize(dataset.data.size());
    size_t col_index_offset = 0;
    bool valid = true;

    for (size_t t=0; t<dataset.data.size(); t++) {
        const auto& trial_matrix = dataset.data[t];
        size_t n_points = trial_matrix.cols();
        size_t n_eval_points = n_points - 2;

        gradients[t] = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(n_vars, n_eval_points);

        try {
            for (size_t v=0; v<n_vars; v++) {
                for (size_t p=1; p<n_points-1; p++) {
                    Eigen::VectorXd point_values = trial_matrix.col(p);
                    gradients[t](v, p-1) = evaluator.evaluate_gradient(v, point_values);
                }
            }
        } catch (const NotImplementedError& e) {
            valid = false;
            break;
        }

        col_index_offset += n_eval_points;
    }

    if (!valid) {
        return pd_matrix;
    }

    size_t row_index = 0;
    for (size_t i=0; i<n_vars; i++) {
        for (size_t j=i+1; j<n_vars; j++) {
            if (i != j) [[likely]] {
                size_t col_index = 0;
                for (size_t t=0; t<dataset.data.size(); t++) {
                    const auto& trial_matrix = dataset.data[t];
                    size_t n_points = trial_matrix.cols();
                    for (size_t p=1; p<n_points-1; p++) {
                        double d_i = gradients[t](i, p-1);
                        double d_j = gradients[t](j, p-1);

                        double pd_value;
                        if (std::abs(d_j)>1e-10) [[likely]] {
                            pd_value = -d_j/d_i;
                        } else if (std::abs(d_i) < 1e-10 && std::abs(d_j) < 1e-10) {
                            pd_value = 0.0;
                        } else {
                            pd_value = std::numeric_limits<double>::quiet_NaN();
                        }

                        if (!std::isinf(std::abs(pd_value)) && !std::isnan(pd_value)) [[likely]] {
                            pd_matrix(row_index, col_index) = pd_value;
                        } else {
                            pd_matrix(row_index, col_index) = 0.0;
                        }

                        col_index++;
                    }
                }
                row_index++;
            }
        }
    }

    return pd_matrix;
}



// for higher-dimensional systems (i.e. 3-4 variables)

/* in physical systems, we assume existence of variable interdependencies,
 * and so we account for these in symbolically calculating paired-partial
 * derivatives */

std::vector<std::vector<int>> get_pairs(int n) {
    std::vector<std::vector<int>> result;
    if (n == 3) {
        // return all three-variable pairings
        result = {
            {0, 1},
            {0, 2},
            {1, 2}
        };
    }
    else if (n == 4) {
        // return all four-variable pairings
        result = {
            {0, 1, 2, 3},
            {0, 2, 1, 3},
            {0, 3, 1, 2}
        };
    }
    return result;
}

inline size_t get_pd_index(size_t var_one, size_t var_two, size_t n_vars) {
    if (var_one > var_two) {
        std::swap(var_one, var_two);
    }
    return var_one * (2 * n_vars - var_one - 1) / 2 + (var_two - var_one - 1);
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> chain_rule(const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::vector<int> &interdependent_pairings, std::unordered_map<int, RCP<const Basic>> &gradient) {
    size_t n_vars = dataset.var_list.size();
    size_t n_rows = n_vars * (n_vars - 1) / 2;
    size_t n_columns = dataset.pd_matrix_columns;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pd_matrix(n_rows, n_columns);
    pd_matrix.setZero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> copy_pd_matrix = pd_matrix;

    UnsupervisedEvaluator evaluator;
    evaluator.prepare_gradient(gradient);

    std::vector<bool> var_deps(n_vars, false);
    std::vector<int> other_indices(n_vars, -1);

    if (n_vars == 4) {
        std::fill(var_deps.begin(), var_deps.end(), true);
        other_indices[interdependent_pairings[0]] = interdependent_pairings[1];
        other_indices[interdependent_pairings[1]] = interdependent_pairings[0];
        other_indices[interdependent_pairings[2]] = interdependent_pairings[3];
        other_indices[interdependent_pairings[3]] = interdependent_pairings[2];
    } else if (n_vars == 3) {
        var_deps[interdependent_pairings[0]] = true;
        var_deps[interdependent_pairings[1]] = true;
        other_indices[interdependent_pairings[0]] = interdependent_pairings[1];
        other_indices[interdependent_pairings[1]] = interdependent_pairings[0];
    }

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> gradients;
    gradients.resize(dataset.data.size());
    bool valid = true;
    size_t col_index_offset = 0;

    for (size_t t=0; t<dataset.data.size(); t++) {
        const auto& trial_matrix = dataset.data[t];
        size_t n_points = trial_matrix.cols();
        size_t n_evaluations = n_points - 2;

        gradients[t] = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(n_vars, n_evaluations);

        try {
            for (size_t v=0; v<n_vars; v++) {
                for (size_t p=1; p<n_points-1; p++) {
                    Eigen::VectorXd point_values = trial_matrix.col(p);
                    gradients[t](v, p-1) = evaluator.evaluate_gradient(v, point_values);
                }
            }
        } catch (const NotImplementedError& e) {
            valid = false;
            break;
        }

        col_index_offset += n_evaluations;
    }

    if (!valid) {
        return pd_matrix;
    }

    size_t row_index = 0;
    for (size_t i=0; i<n_vars; i++) {
        for (size_t j=i+1; j<n_vars; j++) {
            if (i != j) [[likely]] {
                bool var_i_dep = var_deps[i];
                bool var_j_dep = var_deps[j];
                int xi_other_idx = other_indices[i];
                int xj_other_idx = other_indices[j];

                size_t col_index = 0;
                for (size_t t=0; t<dataset.data.size(); t++) {
                    const auto& trial_matrix = dataset.data[t];
                    size_t n_points = trial_matrix.cols();
                    for (size_t p=1; p<n_points-1; p++) {
                        double d_i, d_j;
                        if (var_i_dep) {
                            size_t pd_i_index = get_pd_index(xi_other_idx, i, n_vars);
                            double dp_i = xi_other_idx > i ? 1.0 / numeric_partials(pd_i_index, col_index) : numeric_partials(pd_i_index, col_index);
                            d_i = gradients[t](i, p-1) + gradients[t](xi_other_idx, p-1)*dp_i;
                        } else {
                            d_i = gradients[t](i, p-1);
                        }
                        if (var_j_dep) {
                            size_t pd_j_index = get_pd_index(xj_other_idx, j, n_vars);
                            double dp_j = xj_other_idx > j ? 1.0 / numeric_partials(pd_j_index, col_index) : numeric_partials(pd_j_index, col_index);
                            d_j = gradients[t](j, p-1) + gradients[t](xj_other_idx, p-1)*dp_j;
                        } else {
                            d_j = gradients[t](j, p-1);
                        }

                        double pd_value = 0.0;

                        if (std::abs(d_i) >= 1e-10 && std::abs(d_j) >= 1e-10) {
                            double temp = -d_j/d_i;
                            if (temp == temp && std::isfinite(temp)) {
                                pd_value = temp;
                            }
                        }

                        pd_matrix(row_index, col_index) = pd_value;
                        col_index++;
                    }
                }
                row_index++;
            }
        }
    }

    return pd_matrix;
}

// find paired-partial derivative matrix for each variable interdependency
std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> multi_sym_pds(const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::unordered_map<int, RCP<const Basic>> &gradient) {
    size_t n_vars = dataset.var_list.size();

    std::vector<std::vector<int>> pairs = get_pairs(n_vars);

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> pd_matrix_vector;
    pd_matrix_vector.reserve(pairs.size());

    for (auto& pairing : pairs) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pair_pd_matrix = chain_rule(dataset, numeric_partials, pairing, gradient);
        pd_matrix_vector.push_back(pair_pd_matrix);
    }

    return pd_matrix_vector;
}

