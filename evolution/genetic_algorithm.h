
#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include "evolution/genetic_operations.h"

// probability bounds for genetic operations
struct ProbabilityBounds {
    double min_prob;
    double max_prob;
    double base_prob;
};

const ProbabilityBounds CROSSOVER_BOUNDS = {0.05, 0.6, 0.6};
const ProbabilityBounds SP_BOUNDS = {0.1, 0.15, 0.1};
const ProbabilityBounds MUTATION_BOUNDS = {0.3, 0.8, 0.3};

double calculate_probability_adjustment(const std::deque<double> &fitness_history, double base, double min, double max, bool is_crossover=false);
double crossover_prob(const std::deque<double> &best_fitness_history, int plateau_counter);
double sn_prob(const std::deque<double> &best_fitness_history, int plateau_counter);
double mutate_prob(const std::deque<double> &best_fitness_history, int plateau_counter);

Tree copy_tree(const Tree &tree);
void optimize(std::vector<FFPair> &initial_candidates, const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::deque<double> &best_fitness_history, int &plateau, NodeWeights &island_weights);


#endif //GENETIC_ALGORITHM_H
