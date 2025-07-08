
#include "motif_finder.h"

MotifFinder::MotifFinder(std::size_t M, std::size_t d) {
    M_f.resize(d);
    for (auto &r : M_f) {
        r.reserve(M);
    }
}

size_t MotifFinder::get_motif_index(const Tree &tree) const {
    if (!tree) return 0;
    std::stack<Tree> stack;
    stack.push(tree);
    while (!stack.empty()) {
        Tree n = stack.top();
        stack.pop();
        if (!n) continue;
        if (n->type == NodeType::Variable) {
            return n->var_index;
        }
        switch (n->type) {
            case NodeType::Operation: {
                if (n->right) stack.push(n->right);
                if (n->left) stack.push(n->left);
                break;
            }
            case NodeType::Function: {
                if (n->child) stack.push(n->child);
                break;
            }
            default:
                break;
        }
    }
    return 0;
}

bool is_internal(const Tree &t) {
    if (!t) return false;
    return t->type == NodeType::Variable || t->type == NodeType::Constant;
}

void MotifFinder::find_motifs(std::vector<std::vector<FFPair>> &islands, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_pds, const DataSet &dataset, std::size_t M) {
    std::vector<Motif> motif_batch;
    std::mutex motif_mutex;

    #pragma omp parallel
    {
        std::vector<Motif> local_batch;
        #pragma omp for
        for (size_t i=0; i<islands.size(); i++) {
            for (size_t m=0; m<std::min(M, islands[i].size()); m++) {
                try {
                    Tree tree = sym_to_tree(islands[i][m].function);
                    if (!tree || !tree->left || !tree->right) continue;

                    double tau_fitness = islands[i][m].fitness;
                    RCP<const Basic> left_exp = tree_to_sym(tree->left);
                    RCP<const Basic> right_exp = tree->op == OpType::Subtract ? neg(tree_to_sym(tree->right)) : tree_to_sym(tree->right);

                    double f_L_score = tau_fitness - fitness(right_exp, dataset, numeric_pds), f_R_score = tau_fitness - fitness(right_exp, dataset, numeric_pds);
                    f_L_score /= std::abs(tau_fitness);
                    f_R_score /= std::abs(tau_fitness);

                    // add high-impact motifs to local batch
                    if (f_L_score > 0 && !is_a_Number(*left_exp)) {
                        if (parsimony(tree->left) >= 3) {
                            local_batch.push_back(Motif(tree, tree->left, left_exp, f_L_score));
                        }
                    }
                    if (f_R_score > 0 && !is_a_Number(*right_exp)) {
                        if (parsimony(tree->right) >= 3) {
                            local_batch.push_back(Motif(tree, tree->right, right_exp, f_R_score));
                        }
                    }
                } catch (const std::exception& e) {
                    #pragma omp critical
                    {
                        std::cerr << "ERROR: exception in find_motifs: " << e.what() << std::endl;
                    }
                    continue;
                } catch (...) {
                    continue;
                }
            }
        }

        if (!local_batch.empty()) {
            std::lock_guard<std::mutex> lock(motif_mutex);
            motif_batch.insert(motif_batch.end(), local_batch.begin(), local_batch.end());
        }
    }

    // update motif library
    std::sort(motif_batch.begin(), motif_batch.end(), [](const Motif &a, const Motif &b) {
        return a.f_delta > b.f_delta;
    });

    if (M_f.empty()) {
        size_t d = std::max<size_t>(var_list.size(), 10);
        M_f.resize(d);
    }

    for (const auto &motif : motif_batch) {
        if (motif.f_delta <= 0) continue;
        size_t idx = get_motif_index(motif.subtree);
        bool exists = false;
        for (const auto &existing : M_f[idx]) {
            RCP<const Basic> exp_one = simplify(expand(motif.subexpression));
            RCP<const Basic> exp_two = simplify(expand(existing.subexpression));
            if (eq(*exp_one, *exp_two)) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            M_f[idx].push_back(motif);
            std::sort(M_f[idx].begin(), M_f[idx].end(), [](const Motif &a, const Motif &b) {
                return a.f_delta > b.f_delta;
            });

            const size_t MOTIF_PER_VAR = 10;
            if (M_f[idx].size() > MOTIF_PER_VAR) {
                M_f[idx].erase(M_f[idx].begin() + MOTIF_PER_VAR, M_f[idx].end());
            }
        }
    }
}


void MotifFinder::evolve_motif_population(const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_pds) {
    // build expressions from M_f entries
    const int BATCH_SIZE = 100;
    const int MAX_POPULATION_SIZE = 25;
    std::vector<FFPair> motif_batch;
    std::mutex motif_mutex;
    std::uniform_int_distribution<size_t> M_f_index(0,10);

    bool valid_M_f = false;
    for (const auto &m_v : M_f) {
        if (!m_v.empty()) {
            valid_M_f = true;
            break;
        }
    }
    if (!valid_M_f) return;

    #pragma omp parallel
    {
        std::vector<FFPair> local_batch;
        thread_local std::mt19937 thread_gen(std::random_device{}() + omp_get_thread_num());
        std::uniform_int_distribution<size_t> op_dist(0, 2);
        #pragma omp for
        for (int i=0; i<BATCH_SIZE; i++) {
            try {
                std::vector<size_t> available_vars;
                for (size_t v=0; v<M_f.size(); v++) {
                    if (!M_f[v].empty()) {
                        available_vars.push_back(v);
                    }
                }

                if (available_vars.size() < 2) {
                    continue;
                }

                thread_local std::mt19937 thread_gen(std::random_device{}() + omp_get_thread_num());
                std::uniform_int_distribution<size_t> var_selector(0, available_vars.size() - 1);
                size_t var_one_idx = available_vars[var_selector(thread_gen)];
                size_t var_two_idx = available_vars[var_selector(thread_gen)];

                if (var_one_idx == var_two_idx && available_vars.size() > 1) {
                    do {
                        var_two_idx = available_vars[var_selector(thread_gen)];
                    } while (var_two_idx == var_one_idx);
                }

                // select random motifs for each variable
                std::uniform_int_distribution<size_t> motif1_selector(0, M_f[var_one_idx].size() - 1);
                std::uniform_int_distribution<size_t> motif2_selector(0, M_f[var_two_idx].size() - 1);

                const Motif &motif_one = M_f[var_one_idx][motif1_selector(thread_gen)];
                const Motif &motif_two = M_f[var_two_idx][motif2_selector(thread_gen)];

                size_t op_choice = op_dist(thread_gen);
                RCP<const Basic> combined_expr;
                switch (op_choice) {
                    case 0: {
                        combined_expr = add(motif_one.subexpression, motif_two.subexpression);
                        break;
                    }
                    case 1: {
                        combined_expr = sub(motif_one.subexpression, motif_two.subexpression);
                        break;
                    }
                    case 2: {
                        combined_expr = mul(motif_one.subexpression, motif_two.subexpression);
                        break;
                    }
                    default: {
                        combined_expr = add(motif_one.subexpression, motif_two.subexpression);
                        break;
                    }
                }

                combined_expr = simplify(expand(combined_expr));
                if (is_a_Number(*combined_expr)) continue;

                double fitness_val;
                try {
                    fitness_val = fitness(combined_expr, dataset, numeric_pds);

                    if (!std::isnan(fitness_val) && !std::isinf(fitness_val) && fitness_val < 0) {
                        local_batch.emplace_back(combined_expr, fitness_val);
                    }
                } catch (...) {
                    continue;
                }
            } catch (...) {
                continue;
            }
        }

        if (!local_batch.empty()) {
            std::lock_guard<std::mutex> lock(motif_mutex);
            motif_batch.insert(motif_batch.end(), local_batch.begin(), local_batch.end());
        }
    }

    P_M.insert(P_M.end(), motif_batch.begin(), motif_batch.end());
    std::sort(P_M.begin(), P_M.end(), [](const FFPair &a, const FFPair &b) {
        return a.fitness > b.fitness;
    });
    P_M.resize(MAX_POPULATION_SIZE);
}


void MotifFinder::distribute_motifs(std::vector<std::vector<FFPair> > &islands) {
    if (P_M.empty() || islands.empty()) return;
    std::uniform_int_distribution<size_t> island_idx(0,islands.size()-1);
    const int ISLAND_POPULATION = 400;

    #pragma omp parallel
    {
        thread_local std::mt19937 thread_gen(std::random_device{}() + omp_get_thread_num());
        #pragma omp for
        for (size_t i=0; i<P_M.size(); i++) {
            size_t target_island = island_idx(thread_gen);
            #pragma omp critical
            {
                islands[target_island].emplace_back(P_M[i]);
            }
        }
    }

    #pragma omp parallel for
    for (size_t i=0; i<islands.size(); i++) {
        std::sort(islands[i].begin(), islands[i].end(), [](const FFPair &a, const FFPair &b) {
            return a.fitness > b.fitness;
        });
        islands[i].resize(400);
    }
}

const std::vector<std::vector<Motif> > &MotifFinder::get_library() const {
    return M_f;
}

const std::vector<FFPair> &MotifFinder::get_population() const {
    return P_M;
}

