
#include "fitness.h"

double fitness(const RCP<const Basic>& exp, const DataSet& dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& numeric_partials) {
    double fitness_score;

    if (dataset.is_supervised()) {
        fitness_score = supervised_fitness(exp, dataset);
    } else {
        fitness_score = unsupervised_fitness(exp, dataset, numeric_partials);
    }

    return fitness_score;
}


double supervised_fitness(const RCP<const Basic> &exp, const DataSet &dataset) {
    size_t n_vars = dataset.var_list.size();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data = dataset.data[0];

    Tree expression_tree = sym_to_tree(exp);
    SupervisedEvaluator<double> evaluator(expression_tree, data, n_vars);
    double fitness_score = evaluator.evaluate_tree(data);

    return fitness_score;
}


double unsupervised_fitness(const RCP<const Basic> &exp, const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials) {
    if (is_a_Number(*exp)) [[unlikely]] {
        return std::numeric_limits<double>::lowest();
    }

    size_t n_vars = dataset.var_list.size();
    size_t n_rows = n_vars * (n_vars - 1) / 2;
    size_t n_columns = numeric_partials.cols();

    std::unordered_map<int, RCP<const Basic>> gradient;
    try {
        for (size_t var=0; var<n_vars; var++) {
            gradient[var] = diff(exp, symbol(dataset.var_list[var]));
        }
    } catch (const NotImplementedError& e) {
        return std::numeric_limits<double>::lowest();;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> fitness_matrix(n_rows, n_columns);

    if (n_vars == 2) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sym_partials = sym_pds(dataset, gradient);
        for (size_t row_idx=0; row_idx<n_rows; row_idx++) {
            for (size_t col_idx=0; col_idx<n_columns; col_idx++) {
                fitness_matrix(row_idx, col_idx) = std::log(1.0 + std::abs(sym_partials(row_idx, col_idx) - numeric_partials(row_idx, col_idx)));
            }
        }
    } else if (n_vars > 2) {
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> sym_partials = multi_sym_pds(dataset, numeric_partials, gradient);
        double worst_fitness = std::numeric_limits<double>::max();

        /* we choose the worst-performing variable interdependency. This
         * is because in real physical systems, some candidate expressions
         * may accurately model the system dynamics between a certain pair
         * of variables, but not all of the variables */

        for (const auto &sym : sym_partials) {
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pd_matrix(n_rows, n_columns);
            for (size_t row_idx=0; row_idx<n_rows; row_idx++) {
                for (size_t col_idx=0; col_idx<n_columns; col_idx++) {
                    double num_val = std::abs(numeric_partials(row_idx, col_idx));
                    double sym_val = std::abs(sym(row_idx, col_idx));
                    pd_matrix(row_idx, col_idx) = std::log(1.0 + std::abs(num_val - sym_val));
                }
            }
            double sym_fitness = -1.0 / pd_matrix.size() * pd_matrix.sum();
            if (sym_fitness < worst_fitness) {
                fitness_matrix = pd_matrix;
                worst_fitness = sym_fitness;
            }
        }
    }

    double fitness_score = -1.0 / fitness_matrix.size() * fitness_matrix.sum();

    return fitness_score;
}