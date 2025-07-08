
#ifndef BUILD_CANDIDATES_H
#define BUILD_CANDIDATES_H

#include "initialize_model.h"

using namespace SymEngine;

// starting Node weights
extern std::vector<double> starting_weights;
extern std::vector<bool> allowed_starting_weights;

// === EXPRESSION TREE OPERATORS ===

enum class NodeType {
    Constant,
    Variable,
    Operation,
    Function
};

enum class OpType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power
};

enum class FuncType {
    Sin,
    Sinh,
    aSin,
    Cos,
    Cosh,
    aCos,
    Tan,
    Tanh,
    aTan,
    Log,
    Ln,
    Exp,
    Sqrt
};


// === BINARY EXPRESSION TREES ===

// defines structure for binary expression trees
struct Node;
using Tree = std::shared_ptr<Node>;

// Tree Node structure
struct Node {
    NodeType type;
    OpType op;
    FuncType func;

    double c_value;
    bool normal_const = true;
    int var_index;
    bool in_func;

    Tree left;
    Tree right;
    Tree child;

    // define constructors
    Node (OpType operation, Tree left, Tree right) : type(NodeType::Operation), op(operation), in_func(false), left(left), right(right) {}
    Node (FuncType function, Tree child) : type(NodeType::Function), func(function), child(child) {
        if (function != FuncType::Sqrt) {
            set_func_flag(child);
        }
    }
    Node (int var_idx) : type(NodeType::Variable), var_index(var_idx), in_func(false) {}
    Node(double c, bool is_normal=true) : type(NodeType::Constant), c_value(c), normal_const(is_normal), in_func(false) {}

    // determine if in function Node
private:
    static void set_func_flag(Tree &node) {
        if (!node) {
            return;
        }
        node->in_func = true;
        if (node->left) {
            set_func_flag(node->left);
        }
        if (node->right) {
            set_func_flag(node->right);
        }
        if (node->child) {
            set_func_flag(node->child);
        }
    }
};


// === DIFFERENT PAIRING STRUCTURES ===

// define function/fitness pairing structure
struct FFPair {
    RCP<const Basic> function;
    double fitness;
    FFPair() : function(nullptr), fitness(std::numeric_limits<double>::lowest()) {}
    FFPair(RCP<const Basic> func, double fitness_val) : function(func), fitness(fitness_val) {}
};

// define Tree/fitness pairing structure
struct TFPair {
    Tree tree;
    double fitness;
    TFPair() : tree(nullptr), fitness(std::numeric_limits<double>::lowest()) {}
    TFPair(Tree tree, double fitness_val) : tree(tree), fitness(fitness_val) {}
};

// define Tree/Tree pairing structure
struct TTPair {
    Tree tree_one;
    Tree tree_two;
    TTPair() : tree_one(nullptr), tree_two(nullptr) {}
    TTPair(Tree tree_one, Tree tree_two) : tree_one(tree_one), tree_two(tree_two) {}
};


// === NODE WEIGHTS STRUCTURE ===

struct NodeWeights {
    std::vector<double> node_weights;
    std::vector<bool> allowed_operators;
    NodeWeights() : node_weights(starting_weights), allowed_operators(allowed_starting_weights) {}
    NodeWeights(std::vector<double> w, std::vector<bool> a_w) : allowed_operators(a_w) {
        node_weights.resize(a_w.size());
        for (size_t w_i=0; w_i<a_w.size(); w_i++) {
            node_weights[w_i] = a_w[w_i] ? w[w_i] : 0.0;
        }
    }

    void update_weights(const std::vector<FFPair> &candidates, int topN, double alpha);
    void collection(const Tree &tree, std::vector<double> &usage);
};


// === EXPRESSION CONVERSION AND VALIDATION ===

// converts Tree expression to SymEngine expression
RCP<const Basic> tree_to_sym(const Tree &tree);
// converts SymEngine expression to Tree expression
Tree sym_to_tree(const RCP<const Basic> &exp);
// measures complexity of Tree expression
int parsimony(Tree &tree);
// checks if all variables are present
bool all_variables_present(const Tree &tree, const std::vector<int> &chosen_vars={});


// === TREE GENERATION ===

// finds variables in Tree expression
void find_variables(const Tree &tree, std::vector<bool> &vars_found, const std::vector<int> &chosen_vars={});
// builds Tree expression
Tree build_tree(int current_depth, int max_depth, std::vector<double> &population_weights=starting_weights, int num_vars=var_list.size(), bool allow_func=true, const std::vector<int> &chosen_vars={});
// validates Tree expression and converts to SymEngine expression
RCP<const Basic> build_exp(int current_depth, int max_depth, std::vector<double> &population_weights=starting_weights, const std::vector<int> &chosen_vars={}, bool allow_func=true);

// find partial derivative of a SymEngine expression
RCP<const Basic> derive(const RCP<const Basic> &exp, const std::string &var);
void print_tree(const Tree &exp);

#endif //BUILD_CANDIDATES_H
