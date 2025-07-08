#include "fitness_core/evaluate_tree.h"


// supervised system evaluations
template<typename T>
Eigen::RowVectorXd SupervisedEvaluator<T>::tree_to_Eigen(const Tree &tree, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &predictors) const {
    const int n_samples = predictors.cols();

    if (!tree) {
        return Eigen::RowVectorXd::Constant(n_samples, 0.0);
    }

    switch (tree->type) {
        case NodeType::Constant:
            return Eigen::RowVectorXd::Constant(n_samples, tree->c_value);
        case NodeType::Variable:
            return predictors.row(tree->var_index);
        case NodeType::Operation: {
            Eigen::RowVectorXd left = tree_to_Eigen(tree->left, predictors);
            Eigen::RowVectorXd right = tree_to_Eigen(tree->right, predictors);
            switch (tree->op) {
                case OpType::Add:
                    return left.array() + right.array();
                case OpType::Subtract:
                    return left.array() - right.array();
                case OpType::Multiply:
                    return left.array() * right.array();
                case OpType::Divide:
                    return left.array() / right.array();
                case OpType::Power:
                    return left.array().pow(right.array());
                default:
                    throw std::runtime_error("ERROR: Unknown operator");
            }
        }
        case NodeType::Function: {
            Eigen::RowVectorXd arg = tree_to_Eigen(tree->child, predictors);
            switch (tree->func) {
                case FuncType::Sin:
                    return arg.array().sin();
                case FuncType::Sinh:
                    return arg.array().sinh();
                case FuncType::aSin:
                    return arg.array().asin();
                case FuncType::Cos:
                    return arg.array().cos();
                case FuncType::Cosh:
                    return arg.array().cosh();
                case FuncType::aCos:
                    return arg.array().acos();
                case FuncType::Tan:
                    return arg.array().tan();
                case FuncType::Tanh:
                    return arg.array().tanh();
                case FuncType::aTan:
                    return arg.array().atan();
                case FuncType::Exp:
                    return arg.array().exp();
                case FuncType::Sqrt:
                    return arg.array().sqrt();
                case FuncType::Ln:
                    return arg.array().log();
                case FuncType::Log:
                    return arg.array().log10();
                default:
                    throw std::runtime_error("ERROR: Unknown function");
            }
        }
        default:
            throw std::runtime_error("ERROR: Unknown tree type");
    }
}

template<typename T>
T SupervisedEvaluator<T>::evaluate_tree(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &data) const {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> predictors = data.topRows(n_vars-1);

    Eigen::RowVectorXd V = tree_to_Eigen(expression_tree, predictors);
    const Eigen::RowVectorXd &Tg = data.row(n_vars-1);

    double mle = -((V - Tg).array().abs() + 1.0).log().mean();

    if (std::isnan(mle)) {
        return static_cast<T>(std::numeric_limits<double>::lowest());
    }

    return static_cast<T>(mle);
}

template class SupervisedEvaluator<double>;


// unsupervised system evaluations

template<typename T>
thread_local typename UnsupervisedEvaluator<T>::ThreadLocalBuffers UnsupervisedEvaluator<T>::tls;

template<typename T>
void UnsupervisedEvaluator<T>::prepare_gradient(const std::unordered_map<int, RCP<const Basic>> &gradient) {
    // prepare empty gradient for numerically computed derivatives
    n_vars = gradient.size();
    gradient_trees.clear();
    gradient_trees.reserve(n_vars);

    // convert gradient to Tree representation for value substitution
    for (size_t d=0; d<n_vars; d++) {
        if (auto it = gradient.find(d); it != gradient.end()) {
            gradient_trees.push_back(sym_to_tree(it->second));
        } else {
            gradient_trees.push_back(nullptr);
        }
    }

    if (!tls.initialized) [[unlikely]] {
        tls.values.resize(n_vars);
        tls.initialized = true;
    }
}

template<typename T>
T UnsupervisedEvaluator<T>::evaluate_tree(const Tree &tree, const Eigen::VectorXd &values) const {
    if (!tree) [[unlikely]] {
        return T{0};
    }

    switch (tree->type) {
        case NodeType::Variable: [[likely]] {
            return evaluate_variable(tree, values);
        }
        case NodeType::Constant: [[likely]] {
            return evaluate_constant(tree);
        }
        case NodeType::Operation: {
            return evaluate_operation(tree, values);
        }
        case NodeType::Function: {
            return evaluate_function(tree, values);
        }
        default: [[unlikely]]
            return T{0};
    }
}

template<typename T>
T UnsupervisedEvaluator<T>::evaluate_operation(const Tree &tree, const Eigen::VectorXd &values) const {
    constexpr auto multiplication = [](T a, T b) {
        return a * b;
    };
    constexpr auto addition = [](T a, T b) {
        return a + b;
    };
    constexpr auto subtraction = [](T a, T b) {
        return a - b;
    };
    constexpr auto power = [](T a, T b) {
        return std::pow(a, b);
    };

    T left = evaluate_tree(tree->left, values);
    T right = evaluate_tree(tree->right, values);

    switch (tree->op) {
        case OpType::Multiply: [[likely]]
            return multiplication(left, right);
        case OpType::Add:
            return addition(left, right);
        case OpType::Subtract:
            return subtraction(left, right);
        case OpType::Power:
            return power(left, right);
        default: [[unlikely]]
            return T{0};
    }
}

template<typename T>
T UnsupervisedEvaluator<T>::evaluate_function(const Tree &tree, const Eigen::VectorXd &values) const {
    T arg = evaluate_tree(tree->child, values);

    switch(tree->func) {
        case FuncType::Sin:
            return std::sin(arg);
        case FuncType::Sinh:
            return std::sinh(arg);
        case FuncType::aSin:
            return std::asin(arg);
        case FuncType::Cos:
            return std::cos(arg);
        case FuncType::Cosh:
            return std::cosh(arg);
        case FuncType::aCos:
            return std::acos(arg);
        case FuncType::Tan:
            return std::tan(arg);
        case FuncType::Tanh:
            return std::tanh(arg);
        case FuncType::aTan:
            return std::atan(arg);
        case FuncType::Log:
            return std::log10(arg);
        case FuncType::Ln:
            return std::log(arg);
        case FuncType::Exp:
            return std::exp(arg);
        case FuncType::Sqrt:
            return std::sqrt(arg);
        default: [[unlikely]]
            return T{0};
    }
}

template<typename T>
T UnsupervisedEvaluator<T>::evaluate_gradient(size_t index, const Eigen::VectorXd &values) const {
    if (index >= gradient_trees.size()) [[unlikely]] {
        return T{0};
    }

    const Tree &tree = gradient_trees[index];
    if (!tree) [[unlikely]] {
        return T{0};
    }

    if (tree->type == NodeType::Constant) [[unlikely]] {
        return evaluate_constant(tree);
    }

    return evaluate_tree(tree, values);
}

template class UnsupervisedEvaluator<double>;
