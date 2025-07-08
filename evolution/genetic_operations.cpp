
#include "evolution/genetic_operations.h"

std::array<MutationWeight, 9> mutation_weights = {MutationWeight(5.0, LowTemp), MutationWeight(5.0, LowTemp), MutationWeight(5.0, LowTemp), MutationWeight(15.0, HighTemp), MutationWeight(15.0, HighTemp), MutationWeight(15.0, HighTemp), MutationWeight(15.0, HighTemp), MutationWeight(5.0, LowTemp), MutationWeight(5.0, LowTemp)}; // operation (L), operand (L), coefficient (L), embed in function (H), change function type (H), remove function (H), replace subtree (H), delete subtree (L), do nothing (L)

void collect(const Tree &tree, std::vector<Tree> &nodes, bool in_trig, bool in_exp) {
    if (!tree) return;

    if (!in_trig && !in_exp && tree->type == NodeType::Operation && tree->op != OpType::Power) {
        nodes.push_back(tree);
    }

    bool is_trig = tree->type == NodeType::Function && (tree->func == FuncType::Sin || tree->func == FuncType::Cos);
    bool is_exp = tree->type == NodeType::Operation && tree->op == OpType::Power;

    if (tree->left) collect(tree->left, nodes, in_trig || is_trig, in_exp || is_exp);
    if (tree->right) collect(tree->right, nodes, in_trig || is_trig, in_exp || is_exp);
    if (tree->child) collect(tree->child, nodes, in_trig || is_trig, in_exp || is_exp);
}

void collect_nodes(Tree &tree, std::vector<Tree> &nodes) {
    if (!tree) return;
    if (tree->type == NodeType::Variable || (tree->type == NodeType::Operation && tree->op != OpType::Power) || !tree->in_func) {
        nodes.push_back(tree);
    }
    if (tree->left) collect_nodes(tree->left, nodes);
    if (tree->right) collect_nodes(tree->right, nodes);
}

std::vector<Tree> get_valid_nodes(const Tree &tree) {
    std::vector<Tree> nodes;
    collect(tree, nodes, false, false);
    return nodes;
}


Tree copy_tree(const Tree &tree) {
    if (!tree) return nullptr;
    try {
        Tree copy;
        switch (tree->type) {
            case NodeType::Constant: {
                copy = std::make_shared<Node>(tree->c_value, tree->normal_const);
                break;
            }
            case NodeType::Variable: {
                copy = std::make_shared<Node>(tree->var_index);
                break;
            }
            case NodeType::Operation: {
                copy = std::make_shared<Node>(tree->op, copy_tree(tree->left), copy_tree(tree->right));
                break;
            }
            case NodeType::Function: {
                copy = std::make_shared<Node>(tree->func, copy_tree(tree->child));
                break;
            }
            default:
                break;
        }
        return copy;
    } catch (const std::exception& e) {
        return nullptr;
    }
}

bool find_parent(Tree &tree, Tree &target, Tree &parent) {
    if (!tree) return false;
    if (tree->left == target || tree->right == target || (tree->type == NodeType::Function && tree->child == target)) {
        parent = tree;
        return true;
    }
    return find_parent(tree->left, target, parent) || find_parent(tree->right, target, parent) || (tree->type == NodeType::Function && find_parent(tree->child, target, parent));
}

void swap_subtrees(Tree &tree_one, Tree &point_one, Tree &tree_two, Tree &point_two) {
    if (!tree_one || !tree_two) return;
    Tree parent_one = nullptr;
    Tree parent_two = nullptr;
    find_parent(tree_one, point_one, parent_one);
    find_parent(tree_two, point_two, parent_two);
    Tree copy_one = copy_tree(point_one);
    Tree copy_two = copy_tree(point_two);

    if (!parent_one) {
        tree_one = copy_two;
    } else {
        if (parent_one->left == point_one) {
            parent_one->left = copy_two;
        } else if (parent_one->right == point_one) {
            parent_one->right = copy_two;
        } else if (parent_one->type == NodeType::Function) {
            parent_one->child = copy_two;
        }
    }

    if (!parent_two) {
        tree_two = copy_one;
    } else {
        if (parent_two->left == point_two) {
            parent_two->left = copy_one;
        } else if (parent_two->right == point_two) {
            parent_two->right = copy_one;
        } else if (parent_two->type == NodeType::Function) {
            parent_two->child = copy_one;
        }
    }
}

void crossover(TTPair &tree_pair) {
    Tree offspring_one = copy_tree(tree_pair.tree_one);
    Tree offspring_two = copy_tree(tree_pair.tree_two);

    const int MAX_ATTEMPTS = 5;
    int n_vars = var_list.size();
    for (int i=0; i<MAX_ATTEMPTS; i++) {
        Tree t_one = copy_tree(offspring_one);
        Tree t_two = copy_tree(offspring_two);

        std::vector<Tree> t_nodes_one = get_valid_nodes(t_one);
        std::vector<Tree> t_nodes_two = get_valid_nodes(t_two);

        if (t_nodes_one.empty() || t_nodes_two.empty()) {
            continue;
        }

        std::uniform_int_distribution<size_t> dist_one(0, t_nodes_one.size()-1);
        std::uniform_int_distribution<size_t> dist_two(0, t_nodes_two.size()-1);

        Tree point_one = t_nodes_one[dist_one(gen)];
        Tree point_two = t_nodes_two[dist_two(gen)];
        swap_subtrees(t_one, point_one, t_two, point_two);

        try {
            auto new_exp_one = tree_to_sym(t_one);
            auto new_exp_two = tree_to_sym(t_two);
            if (parsimony(t_one) <= PARSIMONY_SCALE && parsimony(t_two) <= PARSIMONY_SCALE && !eq(*new_exp_one, *tree_to_sym(offspring_one)) && !eq(*new_exp_two, *tree_to_sym(offspring_two)) && all_variables_present(t_one) && all_variables_present(t_two)) {
                tree_pair.tree_one = t_one;
                tree_pair.tree_two = t_two;
                return;
            }
        } catch (...) {
            continue;
        }
    }
}


// swaps individual nodes between binary expression trees
void sn_crossover(TTPair &tree_pair) {
    Tree offspring_one = copy_tree(tree_pair.tree_one);
    Tree offspring_two = copy_tree(tree_pair.tree_two);
    const int MAX_ATTEMPTS = 3;

    for (int i=0; i<MAX_ATTEMPTS; i++) {
        Tree t_one = copy_tree(offspring_one);
        Tree t_two = copy_tree(offspring_two);

        std::vector<Tree> t_nodes_one, t_nodes_two;
        collect_nodes(t_one, t_nodes_one);
        collect_nodes(t_two, t_nodes_two);

        if (t_nodes_one.empty() || t_nodes_two.empty()) {
            continue;
        }
        std::uniform_int_distribution<size_t> dist_one(0, t_nodes_one.size()-1);
        Tree node_one = t_nodes_one[dist_one(gen)];

        std::vector<Tree> matching_nodes;
        for (auto &node : t_nodes_two) {
            if (node->type == node_one->type) {
                if (node->type == NodeType::Operation) {
                    bool is_one_algebraic = node_one->op == OpType::Add || node_one->op == OpType::Subtract || node_one->op == OpType::Multiply;
                    bool is_two_algebraic = node->op == OpType::Add || node->op == OpType::Subtract || node->op == OpType::Multiply;
                    if (is_one_algebraic && is_two_algebraic) {
                        matching_nodes.push_back(node);
                    }
                } else {
                    matching_nodes.push_back(node);
                }
            }
        }

        if (matching_nodes.empty()) {
            continue;
        }
        std::uniform_int_distribution<size_t> dist_two(0, matching_nodes.size()-1);
        Tree node_two = matching_nodes[dist_two(gen)];

        if (node_one->type == NodeType::Operation) {
            std::swap(node_one->op, node_two->op);
        } else if (node_one->type == NodeType::Variable) {
            std::swap(node_one->var_index, node_two->var_index);
        }

        try {
            auto new_exp_one = tree_to_sym(t_one);
            auto new_exp_two = tree_to_sym(t_two);
            if (!eq(*new_exp_one, *tree_to_sym(offspring_one)) && !eq(*new_exp_two, *tree_to_sym(offspring_two)) && all_variables_present(t_one) && all_variables_present(t_two) && parsimony(t_one) <= PARSIMONY_SCALE && parsimony(t_two) <= PARSIMONY_SCALE) {
                tree_pair.tree_one = t_one;
                tree_pair.tree_two = t_two;
                return;
            }
        } catch (...) {}
    }
}

void find_matching_nodes(NodeType type, std::vector<Tree> &nodes, const Tree &tree) {
    if (!tree) return;
    if (tree->type == type) nodes.push_back(tree);
    if (tree->left) find_matching_nodes(type, nodes, tree->left);
    if (tree->right) find_matching_nodes(type, nodes, tree->right);
    if (tree->child) find_matching_nodes(type, nodes, tree->child);
}

void mutate(TFPair &tree, const DataSet &dataset, NodeWeights &island_weights, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials) {
    Tree original_tree = copy_tree(tree.tree);
    std::vector<TFPair> mutations;

    std::vector<double> weight_values;
    for (const auto &w : mutation_weights) {
        weight_values.push_back(w.weight);
    }
    std::discrete_distribution<int> mutate_dist(weight_values.begin(), weight_values.end());

    std::vector<Tree> nodes;
    collect(tree.tree, nodes, false, false);
    if (nodes.empty()) {
        return;
    }

    std::vector<Tree> check_nodes;

    std::vector<Tree> op_nodes;
    find_matching_nodes(NodeType::Operation, op_nodes, original_tree);
    std::vector<Tree> func_nodes;
    find_matching_nodes(NodeType::Function, func_nodes, original_tree);
    std::vector<Tree> var_nodes;
    find_matching_nodes(NodeType::Variable, var_nodes, original_tree);
    std::vector<Tree> coeff_nodes;
    find_matching_nodes(NodeType::Constant, coeff_nodes, original_tree);

    std::vector<double> function_weights;
    std::vector<FuncType> possible_funcs;
    for (size_t i=0; i<6; i++) {
        if (island_weights.node_weights[i+5] > 0.0) {
            function_weights.push_back(island_weights.node_weights[i+5]);
            possible_funcs.push_back(static_cast<FuncType>(i));
        }
    }

    /* to add some control to the mutation process, we pre-determine
     * the type-mutation, validate the type-mutation for the specific
     * expression Tree, and then apply the type-mutation; since mutation
     * is inherently quite random, we can bias mutation types at certain
     * stages of the evolutionary algorithm (e.g. high "replace subtree"
     * rates at Higher Temperatures and high "coefficient" at Lower
     * Temperatures */

    int choice;
    for (int i=0; i<5; i++) {
        choice = mutate_dist(gen);

        if (choice == 0 || choice == 6) { // operation, replace subtree
            check_nodes = op_nodes;
            if (!check_nodes.empty()) {
                break;
            }
        } else if (choice == 1 || choice == 3) { // operand, embed in function
            check_nodes = var_nodes;
            if (function_weights.empty()) {
                choice = 1;
            }
            if (!check_nodes.empty()) {
                break;
            }
        } else if (choice == 4 || choice == 5) { // change function type, remove function
            check_nodes = func_nodes;
            if (!check_nodes.empty()) {
                break;
            }
        } else if (choice == 7) { // delete subtree
            check_nodes = op_nodes;
            check_nodes.insert(check_nodes.end(), func_nodes.begin(), func_nodes.end());
            if (!check_nodes.empty()) {
                break;
            }
        } else if (choice == 2) { // coefficient
            check_nodes = coeff_nodes;
            if (!check_nodes.empty()) {
                break;
            }
        }
    }

    if (check_nodes.empty()) {
        return;
    }

    const int MAX_ATTEMPTS = 3;
    for (int attempt=0; attempt<MAX_ATTEMPTS; attempt++) {
        Tree mutated_tree = copy_tree(tree.tree);
        std::vector<Tree> valid_nodes;
        Tree target_node = nullptr;

        if (choice == 0 || choice == 6) {
            find_matching_nodes(NodeType::Operation, valid_nodes, mutated_tree);
            if (!check_nodes.empty()) {
                std::uniform_int_distribution<size_t> dist(0, valid_nodes.size()-1);
                target_node = valid_nodes[dist(gen)];
            } else {
                return;
            }
        } else if (choice == 1 || choice == 3) {
            find_matching_nodes(NodeType::Variable, valid_nodes, mutated_tree);
            if (!check_nodes.empty()) {
                std::uniform_int_distribution<size_t> dist(0, valid_nodes.size()-1);
                target_node = valid_nodes[dist(gen)];
            } else {
                return;
            }
        } else if (choice == 4 || choice == 5) {
            find_matching_nodes(NodeType::Function, valid_nodes, mutated_tree);
            if (!check_nodes.empty()) {
                std::uniform_int_distribution<size_t> dist(0, valid_nodes.size()-1);
                target_node = valid_nodes[dist(gen)];
            } else {
                return;
            }
        } else if (choice == 7) {
            std::vector<Tree> temp_op, temp_func;
            find_matching_nodes(NodeType::Operation, temp_op, mutated_tree);
            find_matching_nodes(NodeType::Function, temp_func, mutated_tree);
            valid_nodes = temp_op;
            valid_nodes.insert(valid_nodes.end(), temp_func.begin(), temp_func.end());
            if (!check_nodes.empty()) {
                std::uniform_int_distribution<size_t> dist(0, valid_nodes.size()-1);
                target_node = valid_nodes[dist(gen)];
            } else {
                return;
            }
        } else if (choice == 2) {
            find_matching_nodes(NodeType::Constant, valid_nodes, mutated_tree);
            if (!check_nodes.empty()) {
                std::uniform_int_distribution<size_t> dist(0, valid_nodes.size()-1);
                target_node = valid_nodes[dist(gen)];
            } else {
                return;
            }
        }

        if (valid_nodes.empty() || !target_node) [[unlikely]] {
            return;
        }

        switch (choice) {
            case 0: { // Operation
                if (target_node->op == OpType::Add || target_node->op == OpType::Subtract || target_node->op == OpType::Multiply) {
                    std::vector<OpType> possible_ops = {OpType::Add, OpType::Subtract, OpType::Multiply};
                    possible_ops.erase(std::remove(possible_ops.begin(), possible_ops.end(), target_node->op), possible_ops.end());

                    if (!possible_ops.empty()) {
                        std::uniform_int_distribution<size_t> op_dist(0, possible_ops.size()-1);
                        target_node->op = possible_ops[op_dist(gen)];
                    }

                }

                break;
            }
            case 1: { // Operand
                std::uniform_int_distribution<int> var_dist(0, var_list.size()-1);
                int new_index;

                if (var_list.size() == 1) {
                    goto branch;
                }

                do {
                    new_index = var_dist(gen);
                } while (new_index == target_node->var_index);

                target_node->var_index = new_index;

                break;
            }
            case 2: { // Coefficient
                if (!target_node || target_node->type != NodeType::Constant) {
                    continue;
                }

                bool is_exponent = false;
                Tree parent = nullptr;
                find_parent(mutated_tree, target_node, parent);

                if (parent && parent->type == NodeType::Operation && parent->op == OpType::Power) {
                    is_exponent = true;
                }

                if (is_exponent) {
                    std::uniform_real_distribution<double> factor_dist(0.5, 1.5);
                    double scaled = target_node->c_value * factor_dist(gen);
                    target_node->c_value = static_cast<double>(std::max(1, static_cast<int>(std::round(scaled))));
                } else {
                    std::uniform_real_distribution<double> factor_dist(0.5, 1.5);
                    target_node->c_value = std::round(target_node->c_value * factor_dist(gen)*10.0) / 10.0;
                }

                break;
            }
            case 3: { // Embed in function
                if (target_node->in_func) {
                    break;
                }

                std::discrete_distribution<int> func_choice(function_weights.begin(), function_weights.end());
                int func_gen = func_choice(gen);
                FuncType func = possible_funcs[func_gen];
                Tree func_node = std::make_shared<Node>(func, target_node);

                Tree parent = nullptr;
                find_parent(mutated_tree, target_node, parent);

                if (parent) {
                    if (parent->left == target_node) {
                        parent->left = func_node;
                    } else if (parent->right == target_node) {
                        parent->right = func_node;
                    }
                } else {
                    mutated_tree = func_node;
                }

                break;
            }
            case 4: { // Change function type
                if (possible_funcs.empty() || possible_funcs.size() == 1) {
                    break;
                }

                int current_index;
                switch (target_node->func) {
                    case FuncType::Sin:
                        current_index = 0;
                    case FuncType::Sinh:
                        current_index = 1;
                    case FuncType::aSin:
                        current_index = 2;
                    case FuncType::Cos:
                        current_index = 3;
                    case FuncType::Cosh:
                        current_index = 4;
                    case FuncType::aCos:
                        current_index = 5;
                    case FuncType::Tan:
                        current_index = 6;
                    case FuncType::Tanh:
                        current_index = 7;
                    case FuncType::aTan:
                        current_index = 8;
                    case FuncType::Log:
                        current_index = 9;
                    case FuncType::Ln:
                        current_index = 10;
                    case FuncType::Sqrt:
                        current_index = 11;
                }

                std::uniform_int_distribution<int> func_dist(0, possible_funcs.size()-1);
                int new_index;

                do {
                    new_index = func_dist(gen);
                } while (new_index == current_index);

                target_node->func = possible_funcs[new_index];
                break;
            }
            case 5: { // Remove function
                Tree parent = nullptr;
                find_parent(mutated_tree, target_node, parent);

                if (!parent) {
                    continue;
                }

                if (parent->left == target_node) {
                    parent->left = target_node->child;
                } else if (parent->right == target_node) {
                    parent->right = target_node->child;
                }

                break;
            }
            case 6: { // Replace subtree
                std::vector<int> subtree_vars;
                std::vector<bool> vars_found(var_list.size(), false);
                find_variables(target_node, vars_found);

                for (size_t i=0; i<vars_found.size(); i++) {
                    if (vars_found[i]) {
                        subtree_vars.push_back(i);
                    }
                }
                if (subtree_vars.empty() && !var_list.empty()) {
                    std::uniform_int_distribution<int> choose_var(0, var_list.size()-1);
                    subtree_vars.push_back(choose_var(gen));
                }

                Tree new_subtree;
                while (!all_variables_present(new_subtree, subtree_vars)) {
                    new_subtree = build_tree(0, subtree_vars.size(), island_weights.node_weights, subtree_vars.size(), true, subtree_vars);
                }
                Tree parent = nullptr;
                find_parent(mutated_tree, target_node, parent);

                if (parent) {
                    if (parent->left == target_node) {
                        parent->left = new_subtree;
                    } else if (parent->right == target_node) {
                        parent->right = new_subtree;
                    }
                } else {
                    mutated_tree = new_subtree;
                }

                break;
            }
            case 7: { // Delete subtree
                Tree parent = nullptr;
                find_parent(mutated_tree, target_node, parent);

                if (!parent) {
                    continue;
                }

                Tree target_sibling = nullptr;
                if (parent->left == target_node) {
                    target_sibling = parent->right;
                } else if (parent->right == target_node) {
                    target_sibling = parent->left;
                }

                if (!target_sibling) [[unlikely]] {
                    continue;
                }

                Tree grandparent = nullptr;
                find_parent(mutated_tree, parent, grandparent);

                if (!grandparent) {
                    mutated_tree = target_sibling;
                } else {
                    if (grandparent->left == parent) {
                        grandparent->left = target_sibling;
                    } else if (grandparent->right == parent) {
                        grandparent->right = target_sibling;
                    }
                }

                break;
            }
            case 8: { // Nothing
                branch:
                break;
            }
        }

        // validate new tree
        try {
            RCP<const Basic> mutated_exp = tree_to_sym(mutated_tree);
            mutated_exp = simplify(expand(mutated_exp));
            Tree simplified_mutated_tree = sym_to_tree(mutated_exp);
            if (!simplified_mutated_tree || !all_variables_present(simplified_mutated_tree)) {
                continue;
            }
            if (parsimony(simplified_mutated_tree) > PARSIMONY_SCALE) {
                continue;
            }
            double mutated_fitness = fitness(mutated_exp, dataset, numeric_partials);
            if (std::isnan(mutated_fitness) || std::isinf(mutated_fitness)) {
                continue;
            }
            mutations.push_back(TFPair(simplified_mutated_tree, mutated_fitness));
        } catch (...) {
            continue;
        }
    }

    if (mutations.empty()) {
        return;
    }

    // find best mutation
    auto best_mutation = std::max_element(mutations.begin(), mutations.end(), [](TFPair& a, TFPair& b) {
        return a.fitness < b.fitness;
    });

    tree.tree = best_mutation->tree;
    tree.fitness = best_mutation->fitness;
}

// feedback-based adjustment of mutation weights
void update_mutate_weights(double& temperature) {
    double adjustment = 1.0 - temperature;

    for (MutationWeight& w : mutation_weights) {
        if (w.temp == TempWeight::HighTemp) {
            w.weight = std::clamp(15.0 - 10.0*adjustment, 5.0, 15.0);
        } else if (w.temp == TempWeight::LowTemp) {
            w.weight = std::clamp(5.0 + 10.0*adjustment, 5.0, 15.0);
        }
    }
}
