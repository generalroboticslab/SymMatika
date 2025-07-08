
#include "model.h"

// core run function
void Model::run() {
    gen_count = 0;
    loss_history.clear();

    const int NUM_GENERATIONS = 1000;
    const int NUM_ISLANDS = dataset.is_supervised() ? std::min((int)(2 * dataset.var_list.size()-1), 8) : std::min((int)(2 * dataset.var_list.size()), 8);
    const int PLATEAU_THRESHOLD = 75;
    const int MIGRATION_START = 50;
    const int MIGRATION_INTERVAL = 20;
    const double MIGRATION_RATE = 0.02;

    // termination condition
    double epsilon = dataset.is_supervised() ? -1e-6 : -0.025;

    // initialize dataset and numeric partials
    srand(time(0));
    initialize_variables(dataset);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> num_partials = num_pds(dataset);

    // top performers and numerical partial derivatives matrix
    std::vector<FFPair> top_performers;
    const int TOP_SIZE = 6;
    top_performers.reserve(TOP_SIZE);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> numeric_partials = num_pds(dataset);

    // average fitness in top-part of generation
    std::deque<double> fitness_history;

    // island parameters
    std::vector<std::vector<FFPair>> islands(NUM_ISLANDS);
    std::vector<std::deque<double>> fitness_histories(NUM_ISLANDS);
    std::vector<int> plateaus(NUM_ISLANDS, 0);
    std::vector<double> alphas(NUM_ISLANDS, 0.2);
    std::vector<NodeWeights> island_weights;
    island_weights.reserve(NUM_ISLANDS);
    std::vector<double> init_weights(starting_weights.begin(), starting_weights.end());
    for (int i=0; i<NUM_ISLANDS; i++) {
        island_weights.emplace_back(NodeWeights(init_weights, allowed_ops));
    }

    std::function average_fitness = [](std::vector<FFPair> &functions) -> double {
        if (functions.empty()) {
            return std::numeric_limits<double>::lowest();
        }
        double sum = 0.0;

        for (int i=0; i<functions.size()*0.075; i++) {
            sum += functions[i].fitness;
        }
        sum /= functions.size()*0.075;
        return sum;
    };

    // initialize island populations
    for (int i=0; i<NUM_ISLANDS; i++) {
        islands[i] = generate_candidates(initial_pop, max_depth, dataset, numeric_partials, island_weights[i].node_weights);
        fitness_histories[i].clear();
        fitness_histories[i].push_front(average_fitness(islands[i]));
    }

    // define motif library
    MotifFinder mf = MotifFinder(10, var_list.size());

    // begin evolutionary loop
    for (int generation=1; generation<=NUM_GENERATIONS; generation++) {
        if (generation % 5 == 0) {
            std::cout << "Generation " << generation << ": " << std::endl;
            for (auto &p : top_performers) {
                std::cout << *p.function << ", " << p.fitness << std::endl;
            }
            std::cout << "\n\n";
        }

        // stop function call if requested by user
        if (stop_requested) return;

        // check and reinitialize plateaued islands
        for (int i=0; i<NUM_ISLANDS; i++) {
            if (plateaus[i] >= PLATEAU_THRESHOLD) {
                std::vector<FFPair> top_island_performers;
                int top_island_size = std::min(static_cast<int>(islands[i].size() * 0.1), 50);
                top_island_performers.assign(islands[i].begin(), islands[i].begin() + top_island_size);
                islands[i] = generate_candidates(initial_pop, max_depth, dataset, numeric_partials, island_weights[i].node_weights);

                std::sort(islands[i].begin(), islands[i].end(), [](const FFPair &a, const FFPair &b) {
                    return a.fitness > b.fitness;
                });

                for (int t=0; t<top_island_size && t < islands[i].size(); t++) {
                    islands[i][islands[i].size() - t - 1] = top_island_performers[t];
                }

                plateaus[i] = 0;
                alphas[i] = 0.2;
                fitness_histories[i].clear();
                fitness_histories[i].push_front(average_fitness(islands[i]));
            }
        }

        // evolve each island
        #pragma omp parallel for
        for (int i=0; i<NUM_ISLANDS; i++) {
            try {
                // evolve populations
                optimize(islands[i], dataset, numeric_partials, fitness_histories[i], plateaus[i], island_weights[i]);

                // decay learning rates
                if (generation >= 10) {
                    alphas[i] -= 0.0025;
                    alphas[i] = std::clamp(alphas[i], 0.001, 0.2);

                    // update operation weights vector
                    island_weights[i].update_weights(islands[i], islands[i].size()*0.1, alphas[i]);
                }

                // add average fitnesses to fitness histories
                fitness_histories[i].push_front(average_fitness(islands[i]));

                if (fitness_histories[i].size() >= 2 && std::abs(fitness_histories[i][1] - fitness_histories[i].front()) < 0.0005) {
                    plateaus[i]++;
                } else {
                    plateaus[i] = 0;
                }
            } catch (const SymEngine::NotImplementedError& e) {
                continue;
            } catch (const std::exception& e) {
                continue;
            }
        }

        if (NUM_ISLANDS >= 2 && generation > MIGRATION_START && generation % MIGRATION_INTERVAL == 0) {
            double migrate_rate = std::min(MIGRATION_RATE, MIGRATION_RATE*(generation - MIGRATION_START) / (double)(NUM_GENERATIONS - MIGRATION_START));
            int island_migrants = std::max(1, (int)(500 * migrate_rate));

            // migration
            for (int i=0; i<NUM_ISLANDS; i++) {
                int next_island = (i+1) % NUM_ISLANDS;

                std::uniform_int_distribution<size_t> island_choice(0, islands[i].size()-1);
                std::uniform_int_distribution<size_t> next_island_choice(0, islands[next_island].size()-1);
                std::unordered_set<size_t> island_set;
                std::unordered_set<size_t> next_island_set;

                for (int m=0; m<island_migrants; m++) {
                    size_t island_index, next_island_index;

                    do {
                        island_index = island_choice(gen);
                    } while (island_set.find(island_index) != island_set.end() && island_set.size() < islands[i].size());
                    do {
                        next_island_index = next_island_choice(gen);
                    } while (next_island_set.find(next_island_index) != next_island_set.end() && next_island_set.size() < islands[i].size());

                    island_set.insert(island_index);
                    next_island_set.insert(next_island_index);
                    std::swap(islands[i][island_index], islands[next_island][next_island_index]);
                }

                std::sort(islands[i].begin(), islands[i].end(), [](const FFPair& a, const FFPair& b) {
                    return a.fitness > b.fitness;
                });

                std::sort(islands[next_island].begin(), islands[next_island].end(), [](const FFPair& a, const FFPair& b) {
                    return a.fitness > b.fitness;
                });
            }
        }

        for (int i=0; i<NUM_ISLANDS; i++) {
            for (int p=0; p<TOP_SIZE; p++) {
                if (islands[i][p].fitness > top_performers[-1].fitness || top_performers.size() < 10) {
                    top_performers.push_back(islands[i][p]);
                }
            }
            std::sort(top_performers.begin(), top_performers.end(), [](const auto &a, const auto &b) {
                return a.fitness > b.fitness;
            });
            top_performers.resize(TOP_SIZE);
            if (islands[i][0].fitness > epsilon) goto end_model;
        }

        /* in this step, we extract high-impact motifs from top-
         * performing candidates and combine motifs into composite
         * expressions, where after members of the motif population
         * P_M are distributed into the island populations */

        if (generation>=100) {
            mf.find_motifs(islands, numeric_partials, dataset, 5);
            mf.distribute_motifs(islands);
        }

        // update class members
        gen_count++;
        avg_loss = 0.0;
        for (const auto &dq : fitness_histories) {
            if (!dq.empty()) avg_loss += dq.front();
        }
        avg_loss /= fitness_histories.size();

        loss_history.emplace_back(gen_count, avg_loss);
        if (progress) progress(gen_count, avg_loss);
    }

    end_model:

    // TODO fix sorting mechanism

    // std::sort(top_performers.begin(), top_performers.end(), [](const FFPair &a, const FFPair &b) {
        // if (a.fitness != b.fitness) return a.fitness > b.fitness;
        // Tree a_T = sym_to_tree(a.function), b_T = sym_to_tree(b.function);
        // return parsimony(a_T) < parsimony(b_T);
    // });

    final_candidates.clear();
    std::cout << "=== FINAL CANDIDATES (Copy Below Into LaTeX) ===\n";
    std::cout << "\\begin{table}[ht]\n";
    std::cout << "\\centering\n";
    std::cout << "\\renewcommand{\\arraystretch}{1.2}\n";
    std::cout << "\\begin{tabular}{@{} l r @{}}\n";
    std::cout << "\\toprule\n";
    std::cout << "\\textbf{Equation} & \\textbf{Error} \\\\\n";
    std::cout << "\\midrule\n";
    std::cout << std::fixed << std::setprecision(7);
    for (const auto& c : top_performers) {
        final_candidates.push_back(c);
        std::cout << "$" << latex(*c.function) << "$ & " << c.fitness << " \\\\\n";
    }
    std::cout << "\\bottomrule\n";
    std::cout << "\\end{tabular}\n";
    std::cout << "\\end{table}\n";

    std::cout << "Top performer: " << (dataset.is_supervised() ? dataset.get_target_variable() : "f") << " = " << latex(*top_performers[0].function) << std::endl;
}

void Model::set_progress(std::function<void(size_t, double)> p) {
    progress = std::move(p);
}

std::vector<std::pair<std::string, double> > Model::return_final_candidates() const {
    std::vector<std::pair<std::string,double>> final_C;
    final_C.reserve(final_candidates.size());
    for (const auto &c : final_candidates) {
        final_C.emplace_back((dataset.is_supervised() ? dataset.get_target_variable() : "f") + " = " + latex(*c.function), c.fitness);
    }
    return final_C;
}

void easy_run() {
    std::cout << "=== SymMatika ===\n\n";

    std::string file_name;
    char system_char;
    bool system_type;
    std::vector<std::string> variables;
    std::vector<bool> choose_ops;
    int depth;

    system_start:
    file_name.clear(), variables.clear(), choose_ops.clear();

    // data
    std::cout << "Enter filename(.txt/.csv): ";
    std::cin >> file_name;

    // system type
    do {
        std::cout << "\nSearch for explicit mappings? (y/n): ";
        std::cin >> system_char;
    } while (system_char != std::tolower('y') && system_char != std::tolower('n'));
    system_type = system_char=='y';

    // variables
    std::string var_string;
    if (system_type) {
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "\nEnter input variables separated commas and/or spaces: ";
        std::getline(std::cin, var_string);
        std::replace(var_string.begin(), var_string.end(), ',', ' ');
        std::istringstream iss(var_string);
        std::string token;
        while (iss >> token) {
            variables.push_back(token);
        }

        std::string target;
        std::cout << "\nEnter target variable: ";
        std::cin >> target;
        variables.push_back(target);
    } else {
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "\nEnter input variables separated commas and/or spaces: ";
        std::getline(std::cin, var_string);
        std::replace(var_string.begin(), var_string.end(), ',', ' ');
        std::istringstream iss(var_string);
        std::string token;
        while (iss >> token) {
            variables.push_back(token);
        }
    }

    // allowed operators
    const std::vector<std::string> op_names = {
        "add", "sub", "mul", "div", "pow",
        "sin", "sinh", "asin", "cos", "cosh",
        "acos", "tan", "tanh", "atan", "log",
        "ln", "exp", "sqrt"
    };
    std::vector<bool> copy_ops = allowed_starting_weights;
    int select_idx;
    std::cout << "\nAllowed Operators:\nSymMatika supports a number of binary and unary operators. "
                 "Since users may have some understanding or intuition for the system they're trying "
                 "to model,\nfor example pendulum systems involve angles, so trigonometric operators (i.e. cos, sin, tan)"
                 "would be expected to be relevant. We provide the following configurations:\n"
                 "1. Choose algebraic operators only (i.e. +, -, x, /, ^)\n"
                 "2. Choose simple operators only [recommended unless specific system dynamics are known] (i.e. ALGEBRAIC & sin(), cos(), tan(), exp(), sqrt()\n"
                 "3. Choose any operators (individually choose through 18 operators)\n"
                 "4. Choose all operators\n\n";
    do {
        std::cout << "Select any option 1-4: ";
        std::cin >> select_idx;
    } while (select_idx != 1 && select_idx != 2 && select_idx != 3 && select_idx != 4);
    switch (select_idx) {
        case 1: {
            for (int i=5; i<copy_ops.size(); i++) copy_ops[i] = false;
            break;
        }
        case 2: {
            std::vector<bool> simple_ops = {
                true,
                true,
                true,
                true,
                true,
                true,
                false,
                false,
                true,
                false,
                false,
                true,
                false,
                false,
                false,
                false,
                true,
                true
            };
            copy_ops = simple_ops;
            break;
        }
        case 3: {
            char choice;
            for (int i=0; i<op_names.size(); i++) {
                do {
                    std::cout << "Include " << op_names[i] << "? (y/n): ";
                    std::cin >> choice;
                } while (choice != std::tolower('y') && choice != std::tolower('n'));
                copy_ops[i] = choice == 'y';
            }
            break;
        }
        case 4: {
            break;
        }
    }
    choose_ops = copy_ops;

    // choose depth
    std::cout << "\n\n";
    do {
        std::cout << "Candidate Expression Tree Depth:\nthe maximum depth of binary expression trees; in range of 1-5 in ascending candidate expression complexity.\nSelect a number 1-5: ";
        std::cin >> depth;
    } while (depth < 1 && depth > 5);

    // confirm
    char confirm;
    do {
        std::cout << "\n---Please confirm your settings ---\n\n";
        std::cout << "Filename: " << file_name << "\n\n";
        std::cout << "System type: " << (system_type ? "explicit mappings" : "implicit relations") << "\n\n";
        if (system_type) {
            std::cout << "Input variables: ";
            for (int i=0; i<variables.size()-1; i++) std::cout << variables[i] << " ";
            std::cout << "\nTarget variable: " << variables[variables.size()-1];
        } else {
            std::cout << "Input variables: ";
            for (int i=0; i<variables.size()-1; i++) std::cout << variables[i] << " ";
        }
        std::cout << "\n\nOperators:";
        for (int i=0; i<copy_ops.size(); i++) {
            if (copy_ops[i]) {
                std::cout << " " << op_names[i];
            }
        }
        std::cout << "\n\nProceed with these settings? (y/n): ";
        std::cin >> confirm;
    } while (confirm != std::tolower('y') && confirm != std::tolower('n'));

    if (confirm == 'n') goto system_start;
    std::cout << "\n\n";

    DataSet data = DataSet(file_name, system_type, variables);
    Model symMatika = Model(data, 10000, depth, choose_ops);
    symMatika.run();
}
