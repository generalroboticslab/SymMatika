
#include "evolution/genetic_algorithm.h"

// calculates adaptive probability rates for each GP operation
double calculate_probability_adjustment(const std::deque<double> &fitness_history, double base, double min, double max, bool is_crossover) {
    if (fitness_history.size()<1) {
        return base;
    }
    double g_n, g_0;
    g_n = fitness_history.front();
    g_0 = fitness_history.back();
    double new_prob = is_crossover ? base - std::abs((g_0 - g_n) / g_0)*std::abs(max - min) : base + std::abs((g_0 - g_n) / g_0)*std::abs(max - min);
    return new_prob;
}

double crossover_prob(const std::deque<double> &fitness_history, int plateau_counter) {
    double adjustment = calculate_probability_adjustment(fitness_history, CROSSOVER_BOUNDS.base_prob, CROSSOVER_BOUNDS.min_prob, CROSSOVER_BOUNDS.max_prob, true);
    double new_prob = plateau_counter>=2 ? adjustment - 0.07*plateau_counter : adjustment;
    return std::clamp(new_prob, CROSSOVER_BOUNDS.min_prob, CROSSOVER_BOUNDS.max_prob);
}
double sp_prob(const std::deque<double> &fitness_history, int plateau_counter) {
    double adjustment = calculate_probability_adjustment(fitness_history, SP_BOUNDS.base_prob, SP_BOUNDS.min_prob, SP_BOUNDS.max_prob);
    double new_prob = plateau_counter>=2 ? adjustment + 0.07*plateau_counter : adjustment;
    return std::clamp(new_prob, SP_BOUNDS.min_prob, SP_BOUNDS.max_prob);
}
double mutate_prob(const std::deque<double> &fitness_history, int plateau_counter) {
    double adjustment = calculate_probability_adjustment(fitness_history, MUTATION_BOUNDS.base_prob, MUTATION_BOUNDS.min_prob, MUTATION_BOUNDS.max_prob);
    double new_prob = plateau_counter>=2 ? adjustment + 0.07*plateau_counter : adjustment;
    return std::clamp(new_prob, MUTATION_BOUNDS.min_prob, MUTATION_BOUNDS.max_prob);
}

// transform fitness values using Boltzmann distribution
double boltzmann_fitness(double fitness_value, double &temperature) {
    return std::exp(fitness_value / temperature);
}

// probabilty of accepting worse candidate; simulated annealing
double boltzmann_probability(double& fitness_one, double& fitness_two, double &temperature) {
    return std::exp(-(fitness_one - fitness_two) / temperature);
}


// Boltzmann tournament selection
void boltzmann_selection(std::vector<TFPair> &tree_population, TTPair &pair, double &score_one, double &score_two, int &tree_idx_one, int &tree_idx_two, double &temperature) {
    int tournament_size = 6;
    if (tree_population.size() < tournament_size) {
        tournament_size = tree_population.size();
    }

    std::vector<size_t> indices;
    indices.reserve(tournament_size);

    std::unordered_set<size_t> used_indices;
    std::uniform_int_distribution<size_t> dist(0, tree_population.size()-1);

    while (indices.size() < tournament_size) {
        size_t idx = dist(gen);
        if (used_indices.insert(idx).second) {
            indices.push_back(idx);
        }
    }

    std::vector<double> boltzmann_values;
    double boltzmann_sum = 0.0;

    for (const auto& i : indices) {
        double boltzmann_prob = boltzmann_fitness(tree_population[i].fitness, temperature);
        boltzmann_values.push_back(boltzmann_prob);
        boltzmann_sum += boltzmann_prob;
    }

    std::vector<double> probabilities;
    for (const auto& value : boltzmann_values) {
        probabilities.push_back(value / boltzmann_sum);
    }

    /* we employ a hybrid-method of selection, involving principles
     * from tournament and roulette selection. We compute normalized
     * Boltzmann probabilities of tournament members, and then employ
     * roulette selection. This does the following:
     *   1. In high temperature settings, selection is less restricted,
     *   allowing for deeper exploration
     *   2. In low-temperature settings, selection is more restricted,
     *   allowing for optimization of the strongest candidates */

    std::discrete_distribution<int> selection_dist_one(probabilities.begin(), probabilities.end());
    int pick_one = selection_dist_one(gen);
    size_t first_idx = indices[pick_one];

    indices.erase(indices.begin() + pick_one);
    probabilities.erase(probabilities.begin() + pick_one);

    double new_total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    for (auto& p : probabilities) {
        p /= new_total;
    }

    std::discrete_distribution<int> selection_dist_two(probabilities.begin(), probabilities.end());
    int pick_two = selection_dist_two(gen);
    size_t second_idx = indices[pick_two];

    pair.tree_one = tree_population[first_idx].tree;
    pair.tree_two = tree_population[second_idx].tree;

    score_one = tree_population[first_idx].fitness;
    score_two = tree_population[second_idx].fitness;

    tree_idx_one = first_idx;
    tree_idx_two = second_idx;

    if (pair.tree_one == pair.tree_two) {
        pair.tree_two = copy_tree(pair.tree_one);
    }
}


void optimize(std::vector<FFPair> &initial_candidates, const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::deque<double> &best_fitness_history, int &plateau, NodeWeights &island_weights) {
    std::vector<FFPair> population = initial_candidates;
    std::vector<FFPair> new_population;
    new_population.reserve(population.size());
    std::vector<FFPair> offspring;

    // calculate adaptive probabilities
    double p_crossover = crossover_prob(best_fitness_history, plateau);
    double p_sn_crossover = sp_prob(best_fitness_history, plateau);
    double p_mutation = mutate_prob(best_fitness_history, plateau);
    std::vector<double> ga_operations = {p_crossover, p_sn_crossover, p_mutation};

    std::vector<TFPair> tree_population(population.size());
    for (size_t i=0; i<population.size(); i++) {
        tree_population[i] = {sym_to_tree(population[i].function), population[i].fitness};
    }

    // Boltzmann tournament selection
    std::vector<std::pair<TFPair, TFPair>> parents;
    double score_one, score_two;
    int tree_idx_one, tree_idx_two;

    double temperature;
    for (int k=0; k<tree_population.size()/2; k++) {
        temperature = std::clamp(1.0 - k / (double)(tree_population.size()/2), 1e-10, 1.0);
        TTPair tree_pair;
        boltzmann_selection(tree_population, tree_pair, score_one, score_two, tree_idx_one, tree_idx_two, temperature);

        std::discrete_distribution<> ga_dist(ga_operations.begin(), ga_operations.end());
        int choice = ga_dist(gen);

        // apply GP operation
        switch (choice) {
            case 0: {
                crossover(tree_pair);
                break;
            }
            case 1: {
                sn_crossover(tree_pair);
                break;
            }
            case 2: {
                TFPair pair_one = TFPair(tree_pair.tree_one, score_one);
                TFPair pair_two = TFPair(tree_pair.tree_two, score_two);
                mutate(pair_one, dataset, island_weights, numeric_partials);
                mutate(pair_two, dataset, island_weights, numeric_partials);
                tree_pair.tree_one = pair_one.tree;
                tree_pair.tree_two = pair_two.tree;
                break;
            }
        }

        /* we validate the new Tree expressions, and replace
         * their corresponding Tree in population if fitness is
         * improved. If not, then we replace the population Tree
         * with a Boltzmann probability */

        // validate first Tree
        try {
            RCP<const Basic> exp_one = tree_to_sym(tree_pair.tree_one);
            double fitness_one = fitness(exp_one, dataset, numeric_partials);
            if (tree_pair.tree_one && all_variables_present(tree_pair.tree_one)) {
                if (!std::isnan(fitness_one) && !std::isinf(fitness_one)) {
                    if (fitness_one > score_one) {
                        population[tree_idx_one] = FFPair(exp_one, fitness_one);
                    } else {
                        std::uniform_real_distribution<double> boltzmann_prob(0.0, 1.0);
                        if (boltzmann_probability(fitness_one, score_one, temperature) <= boltzmann_prob(gen)) {
                            population[tree_idx_one] = FFPair(exp_one, fitness_one);
                        }
                    }
                }
            }
        } catch (...) {
            continue;
        }

        // validate second Tree
        try {
            RCP<const Basic> exp_two = tree_to_sym(tree_pair.tree_two);
            double fitness_two = fitness(exp_two, dataset, numeric_partials);
            if (tree_pair.tree_two && all_variables_present(tree_pair.tree_two)) {
                if (!std::isnan(fitness_two) && !std::isinf(fitness_two)) {
                    if (fitness_two > score_two) {
                        population[tree_idx_two] = FFPair(exp_two, fitness_two);
                    } else {
                        std::uniform_real_distribution<double> boltzmann_prob(0.0, 1.0);
                        if (boltzmann_probability(fitness_two, score_two, temperature) <= boltzmann_prob(gen)) {
                            population[tree_idx_two] = FFPair(exp_two, fitness_two);
                        }
                    }
                }
            }
        } catch (...) {
            continue;
        }
    }

    // sort final candidates
    std:sort(population.begin(),population.end(),[](const FFPair& a, const FFPair& b) {
        return a.fitness > b.fitness;
    });


    int initial_size = initial_candidates.size();

    // replace worst-performers
    if (plateau >= 5) {
        const int NUM_TO_REPLACE = (int) (0.1 * initial_size);
        if (static_cast<int>(initial_size > NUM_TO_REPLACE)) {
            initial_candidates.erase(initial_candidates.end() - NUM_TO_REPLACE, initial_candidates.end());
        }
        while (static_cast<int>(initial_candidates.size()) < initial_size) {
            RCP<const Basic> candidate = build_exp(0, var_list.size(), island_weights.node_weights);
            double candidate_fitness = fitness(candidate, dataset, numeric_partials);
            if (std::isinf(candidate_fitness) || std::isnan(candidate_fitness)) {
                continue;
            }
            initial_candidates.push_back(FFPair(candidate, candidate_fitness));
        }
    } else if (plateau >= 2) {
        const int SMALL_NUM_TO_REPLACE = (int) (0.05 * initial_size);
        if (static_cast<int>(initial_size > SMALL_NUM_TO_REPLACE)) {
            initial_candidates.erase(initial_candidates.end() - SMALL_NUM_TO_REPLACE, initial_candidates.end());
        }
        while (static_cast<int>(initial_candidates.size()) < initial_size) {
            RCP<const Basic> candidate = build_exp(0, var_list.size(), island_weights.node_weights);
            double candidate_fitness = fitness(candidate, dataset, numeric_partials);
            if (std::isinf(candidate_fitness) || std::isnan(candidate_fitness)) {
                continue;
            }
            initial_candidates.push_back(FFPair(candidate, candidate_fitness));
        }
    }

    std::sort(initial_candidates.begin(), initial_candidates.end(), [](const FFPair &a, const FFPair &b) {
        return a.fitness > b.fitness;
    });

    if (initial_candidates.size() > initial_size) {
        initial_candidates.resize(initial_size);
    }

    // update mutation weights
    update_mutate_weights(temperature);

    initial_candidates = population;
}

