
#include "../tree_construction/generate_population.h"

// generates initial population of candidate functions
std::vector<FFPair> generate_candidates(int num_candidates, int max_depth, DataSet &dataset, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials, std::vector<double> &island_weights) {
    std::vector<FFPair> candidates;
    candidates.reserve(num_candidates);
    std::mutex candidates_mutex;
    std::atomic<int> valid_count{0};

    #pragma omp parallel
    {
        std::vector<FFPair> local_candidates;
        local_candidates.reserve(num_candidates / omp_get_num_threads());
        while (valid_count<num_candidates) {
            if (valid_count>=num_candidates) {
                break;
            }
            try {
                RCP<const Basic> candidate = build_exp(0, max_depth, island_weights);
                Tree tree = sym_to_tree(candidate);

                // check for appropriate parsimony and simplicity
                if (parsimony(tree) > PARSIMONY_SCALE || eq(*simplify(candidate), *zero) || is_a_Number(*simplify(candidate))) continue;
                double fitness_value;
                try {
                    fitness_value = fitness(candidate, dataset, numeric_partials);
                    if (std::isnan(fitness_value) || std::isinf(fitness_value) || std::abs(fitness_value) == 0 || fitness_value>0) continue;
                } catch (const SymEngine::NotImplementedError& e) {
                    continue;
                } catch (const std::exception& e) {
                    continue;
                }

                // update local candidates
                local_candidates.emplace_back(candidate, fitness_value);
                int current_count = valid_count++;
                if (current_count >= num_candidates) break;
            } catch (...) {
                continue;
            }
        }

        // merge threads into global vector
        if (!local_candidates.empty()) {
            std::lock_guard<std::mutex> lock(candidates_mutex);
            candidates.reserve(candidates.size() + local_candidates.size());
            for (auto &c : local_candidates) {
                candidates.emplace_back(std::move(c));
            }
        }
    }

    // partial sort top candidates
    std::partial_sort(candidates.begin(), candidates.begin()+std::min(400,static_cast<int>(candidates.size())), candidates.end(), [](const FFPair& a, const FFPair& b) {
        return a.fitness > b.fitness;
    });
    if (candidates.size()>400) candidates.resize(400);

    #pragma omp parallel for
    for (size_t i=0; i<candidates.size(); i++) {
        candidates[i].function = simplify(expand(candidates[i].function));
    }

    std::cout << "Done generating candidates, total candidate count: " << candidates.size() << "\n\n";

    return candidates;
}
