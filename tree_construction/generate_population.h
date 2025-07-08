
#ifndef GENERATE_POPULATION_H
#define GENERATE_POPULATION_H

#include "../fitness_core/fitness.h"

// generates initial population of candidate functions
std::vector<FFPair> generate_candidates(int num_candidates, int max_depth, DataSet &dataset, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::vector<double> &island_weights);

#endif //GENERATE_POPULATION_H
