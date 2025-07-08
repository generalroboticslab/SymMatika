
#ifndef DIFFERENTIATION_H
#define DIFFERENTIATION_H

#include "fitness_core/evaluate_tree.h"

// define mapping of row-index to variable pairing
extern std::unordered_map<int, std::string> variable_pairing_map;
extern size_t pd_matrix_columns;

// function to initialize variable pairing map
void initialize_pairing_map(std::unordered_map<int, std::string> &variable_pairing_map, const DataSet &dataset);

// numerically approximates partial derivative pairs
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> num_pds(const DataSet &dataset);

// for low-dimensional systems (i.e. 2 variables)
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sym_pds(const DataSet &dataset, std::unordered_map<int, RCP<const Basic>> &gradient);

// for higher-dimensional systems (i.e. 3-4 variables)
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> chain_rule(const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::vector<int> &interdependent_pairings, std::unordered_map<int, RCP<const Basic>> &gradient);
std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> multi_sym_pds(const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::unordered_map<int, RCP<const Basic>> &gradient);

#endif //DIFFERENTIATION_H
