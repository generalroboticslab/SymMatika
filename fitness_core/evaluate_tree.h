
#ifndef EVALUATION_H
#define EVALUATION_H

#include "../tree_construction/build_candidates.h"

#include <symengine/llvm_double.h>
#include <symengine/visitor.h>

template<typename T = double>
class Evaluator {
private:
    Tree tree;
    size_t n_vars;

    struct alignas(32) ThreadLocalBuffers {
        Eigen::VectorXd values;
        bool initialized = false;
    };
    static thread_local ThreadLocalBuffers tls;

};


template<typename T = double>
class SupervisedEvaluator {
public:
    SupervisedEvaluator(const Tree &tree, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &data, const size_t num_vars) : expression_tree(tree), data(data), n_vars(num_vars) {}
    ~SupervisedEvaluator() {}

    T evaluate_tree(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &data) const;

private:
    Tree expression_tree;
    size_t n_vars;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;

    Eigen::RowVectorXd tree_to_Eigen(const Tree &tree, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &predictors) const;
};


template<typename T = double>
class UnsupervisedEvaluator {
private:
    std::vector<Tree> gradient_trees;
    size_t n_vars;

    struct alignas(32) ThreadLocalBuffers {
        Eigen::VectorXd values;
        bool initialized = false;
    };
    static thread_local ThreadLocalBuffers tls;

    [[nodiscard]] T evaluate_tree(const Tree &tree, const Eigen::VectorXd &values) const;
    [[nodiscard]] constexpr T evaluate_constant(const Tree &tree) const noexcept {
        return static_cast<T>(tree->c_value);
    }
    [[nodiscard]] constexpr T evaluate_variable(const Tree &tree, const Eigen::VectorXd &values) const noexcept {
        return static_cast<T>(values[tree->var_index]);
    }
    [[nodiscard]] T evaluate_operation(const Tree &tree, const Eigen::VectorXd &values) const;
    [[nodiscard]] T evaluate_function(const Tree &tree, const Eigen::VectorXd &values) const;

public:
    UnsupervisedEvaluator() = default;
    void prepare_gradient(const std::unordered_map<int, RCP<const Basic>> &gradient);
    [[nodiscard]] T evaluate_gradient(size_t index, const Eigen::VectorXd &values) const;
    void clear() {
        gradient_trees.clear();
        n_vars = 0;
    }
};

#endif //EVALUATION_H
