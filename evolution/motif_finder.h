
#ifndef MOTIF_FINDER_H
#define MOTIF_FINDER_H

#include "evolution/genetic_algorithm.h"

struct Motif {
    Tree tau;
    Tree subtree;
    RCP<const Basic> subexpression;
    double f_delta;
    Motif() : tau(nullptr), subtree(nullptr), subexpression(zero), f_delta(0.0) {}
    Motif(const Tree &tau, const Tree &t, const RCP<const Basic> &f, double d) : tau(tau), subtree(t), subexpression(f), f_delta(d) {}
};

class MotifFinder {
public:
    MotifFinder(std::size_t M, std::size_t d);

    // find and evolve motifs
    void find_motifs(std::vector<std::vector<FFPair>> &islands, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_pds, const DataSet &dataset, std::size_t M);
    void evolve_motif_population(const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_pds);

    // distribute motif-composite expressions into the island populations
    void distribute_motifs(std::vector<std::vector<FFPair>> &islands);
    const std::vector<std::vector<Motif>> &get_library() const;
    const std::vector<FFPair> &get_population() const;
private:
    std::vector<std::vector<Motif>> M_f;
    std::vector<FFPair> P_M;
    size_t get_motif_index(const Tree &tree) const;
};

#endif //MOTIF_FINDER_H
