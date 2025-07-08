
#ifndef GENETIC_OPERATIONS_H
#define GENETIC_OPERATIONS_H

#include "../tree_construction/generate_population.h"
#include "evolution/motif_finder.h"

enum TempWeight {
    HighTemp,
    LowTemp
};

// structure for mutation weight
struct MutationWeight {
    double weight;
    TempWeight temp;
    MutationWeight(double w, TempWeight t) : weight(w), temp(t) {}
};

extern std::array<MutationWeight, 9> mutation_weights;

void collect(const Tree &tree, std::vector<Tree> &nodes, bool in_trig, bool in_exp);
void collect_nodes(Tree &tree, std::vector<Tree> &nodes);
std::vector<Tree> get_subtrees(const Tree &tree);
std::vector<Tree> get_valid_nodes(const Tree &tree);
Tree copy_tree(const Tree &tree);

// update mutation weights
void update_mutate_weights(double &temperature);

// genetic operations
void crossover(TTPair &tree_pair);
void sn_crossover(TTPair &tree_pair);
void mutate(TFPair &tree, const DataSet &dataset, NodeWeights &island_weights, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials);


#endif //GENETIC_OPERATIONS_H
