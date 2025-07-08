
#include "../tree_construction/build_candidates.h"

// probability weights vector for operations, in order: ADD, SUB, MUL, DIV, POW, SIN, SINH, ASIN, COS, COSH, ACOS, TAN, TANH, ATAN, LOG, LN, EXP, SQRT
std::vector<double> starting_weights = {5, 5, 5, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
std::vector<bool> allowed_starting_weights = {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true};

// converts Tree expression to SymEngine expression
RCP<const Basic> tree_to_sym(const Tree &tree) {
    if (!tree) [[unlikely]] {
        throw std::runtime_error("ERROR: Null tree node");
    }

    switch (tree->type) {
        case NodeType::Variable: {
            if (tree->var_index >= 0 && static_cast<size_t>(tree->var_index) < var_list.size()) {
                return symbol(var_list[tree->var_index]);
            }
            break;
        }
        case NodeType::Constant: {
            if (!tree->normal_const) {
                return SymEngine::pi;
            }
            if (std::abs(tree->c_value - std::round(tree->c_value)) < 1e-10) {
                return integer(static_cast<int>(std::round(tree->c_value)));
            }
            return real_double(tree->c_value);
        }
        case NodeType::Operation: {
            RCP<const Basic> left_exp = tree_to_sym(tree->left);
            RCP<const Basic> right_exp = tree_to_sym(tree->right);
            switch (tree->op) {
                case OpType::Add:
                    return add(left_exp, right_exp);
                case OpType::Subtract:
                    return sub(left_exp, right_exp);
                case OpType::Multiply:
                    return mul(left_exp, right_exp);
                case OpType::Divide:
                    return SymEngine::div(left_exp, right_exp);
                case OpType::Power:
                    return pow(left_exp, right_exp);
                default:
                    throw std::runtime_error("ERROR: Unknown operation");
            }
        }
        case NodeType::Function: {
            RCP<const Basic> arg = tree_to_sym(tree->child);
            switch (tree->func) {
                case FuncType::Sin:
                    return sin(arg);
                case FuncType::Sinh:
                    return sinh(arg);
                case FuncType::aSin:
                    return asin(arg);
                case FuncType::Cos:
                    return cos(arg);
                case FuncType::Cosh:
                    return cosh(arg);
                case FuncType::aCos:
                    return acos(arg);
                case FuncType::Tan:
                    return tan(arg);
                case FuncType::Tanh:
                    return tanh(arg);
                case FuncType::aTan:
                    return atan(arg);
                case FuncType::Log:
                    return log(arg, integer(10));
                case FuncType::Ln:
                    return log(arg);
                case FuncType::Exp:
                    return exp(arg);
                case FuncType::Sqrt:
                    return sqrt(arg);
                default:
                    throw std::runtime_error("ERROR: Unknown function");
            }
        }
        default:
            throw std::runtime_error("ERROR: Invalid tree node type");
    }
}

// converts SymEngine expression into Tree expression
Tree sym_to_tree(const RCP<const Basic> &exp) {
    if (eq(*exp, *SymEngine::pi)) {
        return std::make_shared<Node>(M_PI, false);
    }
    if (is_a<Symbol>(*exp)) {
        const Symbol &sym = static_cast<const Symbol&>(*exp);
        std::string name = sym.get_name();
        auto it = std::find(var_list.begin(), var_list.end(), name);
        if (it != var_list.end()) {
            int index = std::distance(var_list.begin(), it);
            return std::make_shared<Node>(index);
        }
        throw std::runtime_error("ERROR: Unexpected variable format");
    }
    if (is_a_Number(*exp)) {
        double value = eval_double(*exp);
        return std::make_shared<Node>(value);
    }

    const vec_basic& args = exp->get_args();

    if (is_a<Add>(*exp)) {
        if (args.empty()) {
            return std::make_shared<Node>(0.0);
        }

        if (args.size() == 1) {
            return sym_to_tree(args[0]);
        }

        /* SymEngine's subtract is add(a, mul(-1, b)), so we determine
         * subtraction by looking at negative arguments */

        if (args.size() == 2) {
            RCP<const Basic> t0 = args[0];
            RCP<const Basic> t1 = args[1];
            std::string s0 = t0->__str__();
            std::string s1 = t1->__str__();
            if (s0 > s1) {
                std::swap(t0, t1);
            }

            Tree left = sym_to_tree(t0);

            bool is_negative = false;
            RCP<const Basic> right_term = t1;

            if (is_a<Mul>(*t1)) {
                const vec_basic& mul_args = t1->get_args();
                for (const auto &mul_arg : mul_args) {
                    if (is_a<Integer>(*mul_arg)) {
                        const Integer& int_factor = down_cast<const Integer&>(*mul_arg);
                        if (int_factor.as_int() < 0) {
                            is_negative = true;
                            vec_basic new_mul_args;
                            for (const auto &ma : mul_args) {
                                if (is_a<Integer>(*ma)) {
                                    const Integer& intf = down_cast<const Integer&>(*ma);
                                    if (intf.as_int() == -1) {
                                        continue;
                                    } else if (intf.as_int() < 0) {
                                        new_mul_args.push_back(integer(-intf.as_int()));
                                    } else {
                                        new_mul_args.push_back(ma);
                                    }
                                } else {
                                    new_mul_args.push_back(ma);
                                }
                            }
                            if (new_mul_args.empty()) {
                                right_term = integer(1);
                            } else if (new_mul_args.size() == 1) {
                                right_term = new_mul_args[0];
                            } else {
                                right_term = mul(new_mul_args);
                            }
                            break;
                        }
                    }
                }
            } else if (is_a<Integer>(*t1)) {
                const Integer& int_val = down_cast<const Integer&>(*t1);
                if (int_val.as_int() < 0) {
                    is_negative = true;
                    right_term = integer(-int_val.as_int());
                }
            }

            Tree right = sym_to_tree(right_term);

            if (is_negative) {
                return std::make_shared<Node>(OpType::Subtract, left, right);
            } else {
                return std::make_shared<Node>(OpType::Add, left, right);
            }
        }

        Tree result = sym_to_tree(args[0]);
        for (size_t i=1; i<args.size(); i++) {
            bool is_negative = false;
            RCP<const Basic> this_term = args[i];

            if (is_a<Mul>(*args[i])) {
                const vec_basic& mul_args = args[i]->get_args();
                for (const auto &mul_arg : mul_args) {
                    if (is_a<Integer>(*mul_arg)) {
                        const Integer& int_factor = down_cast<const Integer&>(*mul_arg);
                        if (int_factor.as_int() < 0) {
                            is_negative = true;
                            vec_basic new_mul_args;
                            for (const auto &ma : mul_args) {
                                if (is_a<Integer>(*ma)) {
                                    const Integer& intf = down_cast<const Integer&>(*ma);
                                    if (intf.as_int() == -1) {
                                        continue;
                                    } else if (intf.as_int() < 0) {
                                        new_mul_args.push_back(integer(-intf.as_int()));
                                    } else {
                                        new_mul_args.push_back(ma);
                                    }
                                } else {
                                    new_mul_args.push_back(ma);
                                }
                            }
                            if (new_mul_args.empty()) {
                                this_term = integer(1);
                            } else if (new_mul_args.size() == 1) {
                                this_term = new_mul_args[0];
                            } else {
                                this_term = mul(new_mul_args);
                            }
                            break;
                        }
                    }
                }
            } else if (is_a<Integer>(*args[i])) {
                const Integer& int_val = down_cast<const Integer&>(*args[i]);
                if (int_val.as_int() < 0) {
                    is_negative = true;
                    this_term = integer(-int_val.as_int());
                }
            }

            Tree right = sym_to_tree(this_term);
            if (is_negative) {
                result = std::make_shared<Node>(OpType::Subtract, result, right);
            } else {
                result = std::make_shared<Node>(OpType::Add, result, right);
            }
        }
        return result;
    }

    if (is_a<Mul>(*exp)) {
        if (args.empty()) {
            return std::make_shared<Node>(1.0);
        }

        if (args.size() == 1) {
            return sym_to_tree(args[0]);
        }

        /* Since Log_10(A) in SymEngine is Log(A) / Log(10),
         * we check for division of logarithmic functions */

        RCP<const Basic> log_arg = zero;
        RCP<const Basic> log_coefficient = integer(1);
        bool ln_term = false;
        bool ln10_term = false;

        for (const auto &arg : args) {
            if (is_a<Log>(*arg)) {
                const RCP<const Log> log_term = rcp_static_cast<const Log>(arg);
                const vec_basic& log_args = log_term->get_args();
                if (log_args.size() == 1 && !is_a<Integer>(*log_args[0])) {
                    ln_term = true;
                    log_arg = log_args[0];
                }
            } else if (is_a<Pow>(*arg)) {
                const RCP<const Pow> p = rcp_static_cast<const Pow>(arg);
                if (is_a<Log>(*p->get_base()) && is_a<Integer>(*p->get_exp())) {
                    int exp_val = down_cast<const Integer&>(*p->get_exp()).as_int();
                    if (exp_val == -1) {
                        const RCP<const Log> log_term = rcp_static_cast<const Log>(p->get_base());
                        const vec_basic& log_args = log_term->get_args();
                        if (log_args.size() == 1 && is_a<Integer>(*log_args[0])) {
                            const Integer& log_base = down_cast<const Integer&>(*log_args[0]);
                            if (log_base.as_int() == 10) {
                                ln10_term = true;
                            }
                        }
                    }
                }
            } else if (is_a_Number(*arg)) {
                log_coefficient = mul(log_coefficient, arg);
            }
        }

        if (ln_term && ln10_term && !eq(*log_arg, *zero)) {
            Tree arg_tree = sym_to_tree(log_arg);
            Tree log_tree = std::make_shared<Node>(FuncType::Log, arg_tree);
            if (!eq(*log_coefficient, *integer(1))) {
                Tree coef_tree = sym_to_tree(log_coefficient);
                return std::make_shared<Node>(OpType::Multiply, coef_tree, log_tree);
            }
            return log_tree;
        }

        if (args.size() == 2) {
            Tree left = sym_to_tree(args[0]);

            if (is_a<Pow>(*args[1])) {
                const RCP<const Pow> p = rcp_static_cast<const Pow>(args[1]);
                RCP<const Basic> exp_val = p->get_exp();
                if (is_a_Number(*exp_val)) {
                    RCP<const Number> exp_num = rcp_static_cast<const Number>(exp_val);
                    if (exp_num->is_negative()) {
                        RCP<const Basic> divisor;
                        if (is_a<Integer>(*exp_val) && rcp_static_cast<const Integer>(exp_val)->as_int() == -1) {
                            divisor = p->get_base();
                        } else {
                            RCP<const Basic> positive_exp = neg(exp_val);
                            divisor = pow(p->get_base(), positive_exp);
                        }
                        Tree right = sym_to_tree(divisor);
                        return std::make_shared<Node>(OpType::Divide, left, right);
                    }
                }
            }

            Tree right = sym_to_tree(args[1]);
            return std::make_shared<Node>(OpType::Multiply, left, right);
        }

        Tree result = nullptr;
        bool first_factor = true;
        for (const auto &arg : args) {
            if (first_factor) {
                if (is_a<Pow>(*arg)) {
                    const RCP<const Pow> p = rcp_static_cast<const Pow>(arg);
                    RCP<const Basic> base = p->get_base();
                    RCP<const Basic> exposant = p->get_exp();
                    if (is_a_Number(*exposant)) {
                        double e_d = eval_double(*exposant);
                        if (e_d < 0) {
                            RCP<const Basic> positive_exp;
                            if (is_a<Integer>(*exposant)) {
                                int ival = down_cast<const Integer&>(*exposant).as_int();
                                positive_exp = integer(-ival);
                            } else {
                                positive_exp = neg(exposant);
                            }
                            RCP<const Basic> pos_pow = pow(base, positive_exp);
                            Tree divisor = sym_to_tree(pos_pow);
                            Tree one = std::make_shared<Node>(1.0);
                            result = std::make_shared<Node>(OpType::Divide, one, divisor);
                            first_factor = false;
                            continue;
                        }
                    }
                }
                result = sym_to_tree(arg);
                first_factor = false;
            } else {
                if (is_a<Pow>(*arg)) {
                    const RCP<const Pow> p = rcp_static_cast<const Pow>(arg);
                    RCP<const Basic> base = p->get_base();
                    RCP<const Basic> exposant = p->get_exp();
                    if (is_a_Number(*exposant)) {
                        double e_d = eval_double(*exposant);
                        if (e_d < 0) {
                            RCP<const Basic> positive_exp;
                            if (is_a<Integer>(*exposant)) {
                                int ival = down_cast<const Integer&>(*exposant).as_int();
                                positive_exp = integer(-ival);
                            } else {
                                positive_exp = neg(exposant);
                            }
                            RCP<const Basic> pos_pow = pow(base, positive_exp);
                            Tree divisor = sym_to_tree(pos_pow);
                            result = std::make_shared<Node>(OpType::Divide, result, divisor);
                            continue;
                        }
                    }
                }
                Tree right = sym_to_tree(arg);
                result = std::make_shared<Node>(OpType::Multiply, result, right);
            }
        }
        return result;
    }

    if (is_a<Pow>(*exp)) {
        if (args.size() != 2) {
            return nullptr;
        }

        const RCP<const Pow> pow = rcp_static_cast<const Pow>(exp);
        RCP<const Basic> base = pow->get_base();
        RCP<const Basic> exponent = pow->get_exp();

        if (eq(*base, *E)) {
            Tree arg = sym_to_tree(exponent);
            return std::make_shared<Node>(FuncType::Exp, arg);
        }

        if (is_a<Rational>(*exponent)) {
            const RCP<const Rational> rational = rcp_static_cast<const Rational>(exponent);
            if (rational->is_positive()) {
                double rational_val = eval_double(*rational);
                if (std::abs(rational_val - 0.5) < 1e-10) {
                    Tree arg = sym_to_tree(base);
                    return std::make_shared<Node>(FuncType::Sqrt, arg);
                }
            }
        } else if (is_a<RealDouble>(*exponent)) {
            double exp_val = down_cast<const RealDouble&>(*exponent).as_double();
            if (std::abs(exp_val - 0.5) < 1e-10) {
                Tree arg = sym_to_tree(base);
                return std::make_shared<Node>(FuncType::Sqrt, arg);
            }
        } else if (is_a<Pow>(*base) && is_a<Integer>(*exponent)) {
            const RCP<const Pow> inner_pow = rcp_static_cast<const Pow>(base);
            const RCP<const Basic> inner_base = inner_pow->get_base();
            const RCP<const Basic> inner_exp = inner_pow->get_exp();

            if (is_a<Rational>(*inner_exp)) {
                const RCP<const Rational> inner_rational = rcp_static_cast<const Rational>(inner_exp);
                double inner_rational_val = eval_double(*inner_rational);
                int outer_exp = down_cast<const Integer&>(*exponent).as_int();

                if (inner_rational_val > 0 && inner_rational_val < 1) {
                    double result_exp = outer_exp * inner_rational_val;

                    if (std::abs(result_exp - 1.0) < 1e-10) {
                        return sym_to_tree(inner_base);
                    }

                    if (std::abs(result_exp - 0.5) < 1e-10) {
                        Tree arg = sym_to_tree(inner_base);
                        return std::make_shared<Node>(FuncType::Sqrt, arg);
                    }

                    Tree base_tree = sym_to_tree(inner_base);
                    Tree exp_tree = std::make_shared<Node>(result_exp);
                    return std::make_shared<Node>(OpType::Power, base_tree, exp_tree);
                }
            }
        }

        Tree base_tree = sym_to_tree(base);
        Tree exp_tree = sym_to_tree(exponent);
        return std::make_shared<Node>(OpType::Power, base_tree, exp_tree);
    }

    if (is_a<Sin>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::Sin, arg);
    }

    if (is_a<Sinh>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::Sinh, arg);
    }

    if (is_a<ASin>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::aSin, arg);
    }

    if (is_a<Cos>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::Cos, arg);
    }

    if (is_a<Cosh>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::Cosh, arg);
    }

    if (is_a<ACos>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::aCos, arg);
    }

    if (is_a<Tan>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::Tan, arg);
    }

    if (is_a<Tanh>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::Tanh, arg);
    }

    if (is_a<ATan>(*exp)) {
        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::aTan, arg);
    }

    if (is_a<Log>(*exp)) {
        if (args.empty()) {
            return nullptr;
        }

        if (args.size() > 1) {
            Tree arg = sym_to_tree(args[0]);
            if (is_a<Integer>(*args[1])) {
                const Integer& base = down_cast<const Integer&>(*args[1]);
                if (base.as_int() == 10) {
                    return std::make_shared<Node>(FuncType::Log, arg);
                }
            }
        }

        Tree arg = sym_to_tree(args[0]);
        return std::make_shared<Node>(FuncType::Ln, arg);
    }

    return nullptr;
}


// complexity is evaluated by Node count
int parsimony(Tree &tree) {
    if (!tree) {
        return 0;
    }
    int count = 1;
    if (tree->type == NodeType::Operation) {
        count += parsimony(tree->left) + parsimony(tree->right);
    } else if (tree->type == NodeType::Function) {
        count += parsimony(tree->child);
    }
    return count;
}

void find_variables(const Tree &tree, std::vector<bool> &vars_found, const std::vector<int> &chosen_vars) {
    if (!tree) {
        return;
    }
    if (tree->type == NodeType::Variable) {
        int idx = tree->var_index;
        if (chosen_vars.empty()) {
            if (idx >= 0 && static_cast<size_t>(idx) < vars_found.size()) {
                vars_found[idx] = true;
            }
        } else {
            for (size_t i=0; i<chosen_vars.size(); i++) {
                if (chosen_vars[i] == idx) {
                    vars_found[i] = true;
                    break;
                }
            }
        }
    } else if (tree->type == NodeType::Operation) {
        find_variables(tree->left, vars_found, chosen_vars);
        find_variables(tree->right, vars_found, chosen_vars);
    } else if (tree->type == NodeType::Function) {
        find_variables(tree->child, vars_found, chosen_vars);
    }
}

bool all_variables_present(const Tree &tree, const std::vector<int> &chosen_vars) {
    int var_size = chosen_vars.empty() ? var_list.size() : chosen_vars.size();
    std::vector<bool> vars_found(var_size, false);

    find_variables(tree, vars_found, chosen_vars);
    for (bool var : vars_found) {
        if (!var) {
            return false;
        }
    }
    return true;
}

// build simple argument for functions (i.e. sin, cos, etc.)
Tree build_func_tree(int depth, int num_vars, const std::vector<int> &chosen_vars={}) {
    const int MAX_DEPTH = 2;
    if (depth >= MAX_DEPTH) {
        std::uniform_int_distribution<int> var_dist(0, num_vars-1);
        return chosen_vars.empty() ? std::make_shared<Node>(var_dist(gen)) : std::make_shared<Node>(chosen_vars[var_dist(gen)]);
    }

    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    double choice = prob_dist(gen);
    if (choice<0.3) {
        std::uniform_int_distribution<int> var_dist(0, num_vars-1);
        return chosen_vars.empty() ? std::make_shared<Node>(var_dist(gen)) : std::make_shared<Node>(chosen_vars[var_dist(gen)]);
    }
    if (choice<0.4) {
        std::bernoulli_distribution pi_dist(0.3);
        std::uniform_real_distribution<double> const_dist(-5.0, 5.0);
        double coefficient;
        if (pi_dist(gen)) {
            coefficient = M_PI;
            return std::make_shared<Node>(coefficient, false);
        }
        coefficient = const_dist(gen);
        coefficient = std::round(coefficient * 10.0) / 10.0;
        return std::make_shared<Node>(coefficient, true);
    }

    std::uniform_int_distribution<int> op_dist(0, 3);
    int op_choice = op_dist(gen);
    if (op_choice == 3) {
        Tree base = build_func_tree(depth+1, num_vars, chosen_vars);
        if (base->type == NodeType::Variable || base->type == NodeType::Constant) {
            return std::make_shared<Node>(OpType::Power, base, std::make_shared<Node>(2.0));
        }
        op_choice = 2;
    }

    OpType operation;
    switch(op_choice) {
        case 0:
            operation = OpType::Add;
            break;
        case 1:
            operation = OpType::Subtract;
            break;
        case 2:
            operation = OpType::Multiply;
            break;
        default:
            operation = OpType::Add;
    }

    Tree left = build_func_tree(depth+1, num_vars, chosen_vars);
    Tree right = build_func_tree(depth+1, num_vars, chosen_vars);
    return std::make_shared<Node>(operation, left, right);
}


Tree build_tree(int current_depth, int max_depth, std::vector<double> &population_weights, int num_vars, bool allow_func, const std::vector<int> &chosen_vars) {
    if (current_depth >= max_depth) {
        int var_index;
        if (chosen_vars.empty()) {
            std::uniform_int_distribution<int> var_dist(0, num_vars-1);
            var_index = var_dist(gen);
        } else {
            std::uniform_int_distribution<int> var_dist(0, chosen_vars.size()-1);
            var_index = chosen_vars[var_dist(gen)];
        }
        return std::make_shared<Node>(var_index);
    }

    std::discrete_distribution<int> op_dist(population_weights.begin(), population_weights.end());
    int op_choice = op_dist(gen);

    switch (op_choice) {
        case 0: { // Add
            Tree left = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            Tree right = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            return std::make_shared<Node>(OpType::Add, left, right);
        }

        case 1: { // Sub
            Tree left = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            Tree right = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            return std::make_shared<Node>(OpType::Subtract, left, right);
        }

        case 2: { // Multiply
            Tree left, right;
            std::vector<double> mul_weights = {3, 2, 1};
            std::discrete_distribution<size_t> mul_dist(mul_weights.begin(), mul_weights.end());

            size_t left_choice = mul_dist(gen);
            size_t right_choice;
            do {
                right_choice = mul_dist(gen);
            } while (right_choice == 0);

            switch (left_choice) {
                case 0: { // Coefficient
                    std::bernoulli_distribution pi_dist(0.2);
                    if (pi_dist(gen)) {
                        left = std::make_shared<Node>(M_PI, false);
                    } else {
                        std::uniform_real_distribution<double> const_dist(-10.0, 10.0);
                        double c = std::round(const_dist(gen) * 10.0) / 10.0;
                        left = std::make_shared<Node>(c, true);
                    }
                    break;
                }
                case 1: { // Variable
                    std::uniform_int_distribution<int> var_dist(0, var_list.size()-1);
                    left = std::make_shared<Node>(var_dist(gen));
                    break;
                }
                case 2: { // Subtree
                    left = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
                    break;
                }
            }

            switch (right_choice) {
                case 0: { // Coefficient
                    std::bernoulli_distribution pi_dist(0.2);
                    if (pi_dist(gen)) {
                        right = std::make_shared<Node>(M_PI, /*generic=*/false);
                    } else {
                        std::uniform_real_distribution<double> const_dist(-10.0, 10.0);
                        double c = std::round(const_dist(gen) * 10.0) / 10.0;
                        right = std::make_shared<Node>(c, /*generic=*/true);
                    }
                    break;
                }
                case 1: { // Variable
                    std::uniform_int_distribution<int> var_dist(0, var_list.size()-1);
                    right = std::make_shared<Node>(var_dist(gen));
                    break;
                }
                case 2: { // Subtree
                    right = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
                    break;
                }
            }

            if (left->type == NodeType::Constant && right->type == NodeType::Constant) {
                bool L_is_pi = !left->normal_const;
                bool R_is_pi = !right->normal_const;
                if (!(L_is_pi ^ R_is_pi)) {
                    right = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
                }
            }

            return std::make_shared<Node>(OpType::Multiply, left, right);
        }

        case 3: { // Division
            Tree numerator, denominator;
            std::vector<double> division_weights = {1, 1};
            std::discrete_distribution<int> div_dist(division_weights.begin(), division_weights.end());

            size_t numerator_choice = div_dist(gen);
            size_t denominator_choice;
            do {
                denominator_choice = div_dist(gen);
            } while (denominator_choice == 0);

            switch (numerator_choice) {
                case 0: { // Coefficient
                    std::bernoulli_distribution pi_dist(0.2);
                    if (pi_dist(gen)) {
                        numerator = std::make_shared<Node>(M_PI, false);
                    } else {
                        std::uniform_real_distribution<double> const_dist(-10.0, 10.0);
                        double c = std::round(const_dist(gen) * 10.0) / 10.0;
                        numerator = std::make_shared<Node>(c, true);
                    }
                    break;
                }
                case 1: { // Variable
                    std::uniform_int_distribution<int> var_dist(0, var_list.size() - 1);
                    numerator = std::make_shared<Node>(var_dist(gen));
                    break;
                }
                case 2: { // Subtree
                    numerator = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
                    break;
                }
            }

            switch (denominator_choice) {
                case 0: { // Coefficient
                    std::bernoulli_distribution pi_dist(0.2);
                    if (pi_dist(gen)) {
                        denominator = std::make_shared<Node>(M_PI, /*generic=*/false);
                    } else {
                        std::uniform_real_distribution<double> const_dist(-10.0, 10.0);
                        double c = std::round(const_dist(gen) * 10.0) / 10.0;
                        denominator = std::make_shared<Node>(c, /*generic=*/true);
                    }
                    break;
                }
                case 1: { // Variable
                    std::uniform_int_distribution<int> var_dist(0, var_list.size()-1);
                    denominator = std::make_shared<Node>(var_dist(gen));
                    break;
                }
                case 2: { // Subtree
                    denominator = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
                    break;
                }
            }

            if (numerator->type == NodeType::Constant && denominator->type == NodeType::Constant) {
                bool N_is_pi = !numerator->normal_const;
                bool D_is_pi = !denominator->normal_const;
                if (!(N_is_pi ^ D_is_pi)) {
                    denominator = build_tree(current_depth + 1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
                }
            }

            return std::make_shared<Node>(OpType::Divide, numerator, denominator);
        }

        case 4: { // Power
            std::uniform_real_distribution<double> pow_dist(0.0,1.0);
            double pow_choice = pow_dist(gen);
            Tree exp_node;

            int var_index;
            if (chosen_vars.empty()) {
                std::uniform_int_distribution<int> var_dist(0, num_vars-1);
                var_index = var_dist(gen);
            } else {
                std::uniform_int_distribution<int> var_dist(0, chosen_vars.size()-1);
                var_index = chosen_vars[var_dist(gen)];
            }

            if (pow_choice<=0.99) {
                std::vector<double> exponent_weights = {5.0, 2.0, 1.0, 0.8, 0.6, 0.4};
                std::discrete_distribution<int> exp_dist(exponent_weights.begin(), exponent_weights.end());
                int exp_value = exp_dist(gen) + 2;
                exp_node = std::make_shared<Node>(static_cast<double>(exp_value));
            } else {
                int pow_index;
                if (chosen_vars.empty()) {
                    std::uniform_int_distribution<int> var_dist(0, num_vars-1);
                    pow_index = var_dist(gen);
                } else {
                    std::uniform_int_distribution<int> var_dist(0, chosen_vars.size()-1);
                    pow_index = chosen_vars[var_dist(gen)];
                }
                exp_node = std::make_shared<Node>(pow_index);
            }

            Tree var_node = std::make_shared<Node>(var_index);
            return std::make_shared<Node>(OpType::Power, var_node, exp_node);
        }

        case 5: { // Sin
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Sin, arg);
        }

        case 6: { // Sinh
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Sinh, arg);
        }

        case 7: { // aSin
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::aSin, arg);
        }

        case 8: { // Cos
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Cos, arg);
        }

        case 9: { // Cosh
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Cosh, arg);
        }

        case 10: { // aCos
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::aCos, arg);
        }

        case 11: { // Tan
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Tan, arg);
        }

        case 12: { // Tanh
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Tanh, arg);
        }

        case 13: { // aTan
            if (!allow_func) {
                return build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            }
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::aTan, arg);
        }

        case 14: { // Log
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Log, arg);
        }

        case 15: { // Ln
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Ln, arg);
        }

        case 16: { // Exp
            Tree arg = build_func_tree(0, num_vars, chosen_vars);
            return std::make_shared<Node>(FuncType::Exp, arg);
        }

        case 17: { // Sqrt
            Tree arg = build_tree(current_depth+1, max_depth, population_weights, num_vars, allow_func, chosen_vars);
            return std::make_shared<Node>(FuncType::Sqrt, arg);
        }

        default:
            return std::make_shared<Node>(0.0);
    }
}

// validate Tree and convert to SymEngine
RCP<const Basic> build_exp(int current_depth, int max_depth, std::vector<double> &population_weights, const std::vector<int> &chosen_vars, bool allow_func) {
    int n_vars = chosen_vars.empty() || chosen_vars.size() < 2 ? var_list.size() : chosen_vars.size();
    while (true) {
        try {
            Tree candidate_tree = build_tree(current_depth, max_depth, population_weights, n_vars, allow_func, chosen_vars);
            RCP<const Basic> exp = tree_to_sym(candidate_tree);
            if (is_a<Infty>(*exp) || is_a<NaN>(*exp) || is_a_Number(*exp)) continue;
            try {
                exp = simplify(expand(exp));
                if (is_a<Infty>(*exp) || is_a<NaN>(*exp)) continue;

                Tree simplified_tree = sym_to_tree(exp);
                if (!simplified_tree) continue;
                if (parsimony(simplified_tree) > PARSIMONY_SCALE || !all_variables_present(simplified_tree, chosen_vars)) continue;

                return exp;
            } catch (...) {
                continue;
            }
        } catch (...) {
            continue;
        }
    }
}


/* this function populates a frequency vector usage with Node operation &
 * function frequencies */
void NodeWeights::collection(const Tree &tree, std::vector<double> &usage) {
    if (!tree) {
        return;
    }

    if (tree->type == NodeType::Operation) {
        switch (tree->op) {
            case OpType::Add:
                usage[0] += 1.0;
                break;
            case OpType::Subtract:
                usage[1] += 1.0;
                break;
            case OpType::Multiply: {
                usage[2] += 1.0;
                break;
            }
            case OpType::Divide: {
                usage[3] += 1.0;
                break;
            }
            case OpType::Power:
                if (tree->right && tree->right->type == NodeType::Constant) {
                    double exponent = tree->right->c_value;
                    if (std::fabs(exponent - 2.0) < 1e-12) {
                        usage[4] += 1.0;
                    }
                }
                break;
        }
    }
    else if (tree->type == NodeType::Function) {
        switch (tree->func) {
            case FuncType::Sin: {
                usage[5] += 1.0;
                break;
            }
            case FuncType::Sinh: {
                usage[6] += 1.0;
                break;
            }
            case FuncType::aSin: {
                usage[7] += 1.0;
                break;
            }
            case FuncType::Cos: {
                usage[8] += 1.0;
                break;
            }
            case FuncType::Cosh: {
                usage[9] += 1.0;
                break;
            }
            case FuncType::aCos: {
                usage[10] += 1.0;
                break;
            }
            case FuncType::Tan: {
                usage[11] += 1.0;
                break;
            }
            case FuncType::Tanh: {
                usage[12] += 1.0;
                break;
            }
            case FuncType::aTan: {
                usage[13] += 1.0;
                break;
            }
            case FuncType::Log: {
                usage[14] += 1.0;
                break;
            }
            case FuncType::Ln: {
                usage[15] += 1.0;
                break;
            }
            case FuncType::Exp: {
                usage[16] += 1.0;
                break;
            }
            case FuncType::Sqrt: {
                usage[17] += 1.0;
                break;
            }
        }
    }

    collection(tree->left, usage);
    collection(tree->right, usage);
    if (tree->type == NodeType::Function) {
        collection(tree->child, usage);
    }
}

// update Node weights
void NodeWeights::update_weights(const std::vector<FFPair> &candidates, int topN, double alpha) {
    std::vector<double> usage(node_weights.size(), 0.0);
    topN = std::min(topN, (int)candidates.size());

    for (int i=0; i<topN; i++) {
        auto tree = sym_to_tree(candidates[i].function);
        collection(tree, usage);
    }

    // less emphasis on more common operators like + or -
    static std::vector<double> op_weights = {
        0.2, // +
        0.2, // -
        0.5, // x
        0.5, // ÷
        0.7, // ^
        1.0, // sin()
        1.0, // sinh()
        1.0, // asin()
        1.0, // cos()
        1.0, // cosh()
        1.0, // acos()
        1.0, // tan()
        1.0, // tanh()
        1.0, // atan()
        1.0, // log
        1.0, // ln
        1.0, // exp
        1.0  // sqrt
    };

    for (size_t i=0; i<usage.size(); i++) {
        usage[i]*=op_weights[i];
    }
    for (size_t i=0; i<node_weights.size(); i++) {
        node_weights[i]+=alpha*usage[i];
    }

    // normalize Node weights
    double sum_w = 0.0;
    for (auto w : node_weights) {
        sum_w+=w;
    }
    if (sum_w < 1e-12) {
        double val = 1.0/node_weights.size();
        for (auto &w : node_weights) {
            w = val;
        }
    } else {
        for (auto &w : node_weights) {
            w/=sum_w;
        }
    }
}


std::string tree_to_string(const Tree &tree, std::string &f) {
    if (tree->type == NodeType::Constant) {
        if (std::abs(tree->c_value - std::round(tree->c_value)) < 1e-10) {
            f = std::to_string(static_cast<int>(std::round(tree->c_value)));
        } else {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << tree->c_value;
            f = oss.str();
        }
        return f;
    }

    if (tree->type == NodeType::Variable) {
        if (tree->var_index >= 0 && static_cast<size_t>(tree->var_index) < var_list.size()) {
            f = var_list[tree->var_index];
        }
        return f;
    }

    if (tree->type == NodeType::Operation) {
        std::string left_str, right_str;
        tree_to_string(tree->left, left_str);
        tree_to_string(tree->right, right_str);

        bool left_bracket = tree->left && tree->left->type == NodeType::Operation && ((tree->op == OpType::Multiply || tree->op == OpType::Divide || tree->op == OpType::Power) && (tree->left->op == OpType::Add || tree->left->op == OpType::Subtract));
        bool right_bracket = tree->right && tree->right->type == NodeType::Operation && ((tree->op == OpType::Subtract && (tree->right->op == OpType::Add || tree->right->op == OpType::Subtract)) || (tree->op == OpType::Divide && (tree->right->op == OpType::Add || tree->right->op == OpType::Subtract || tree->right->op == OpType::Multiply || tree->right->op == OpType::Divide)) || (tree->op == OpType::Power && (tree->right->op == OpType::Add || tree->right->op == OpType::Subtract || tree->right->op == OpType::Multiply || tree->right->op == OpType::Divide || tree->right->op == OpType::Power)));

        if (left_bracket) {
            left_str = "(" + left_str + ")";
        }

        if (right_bracket) {
            right_str = "(" + right_str + ")";
        }

        switch (tree->op) {
            case OpType::Add:
                f = left_str + " + " + right_str;
                break;
            case OpType::Subtract:
                f = left_str + " - " + right_str;
                break;
            case OpType::Multiply:
                f = left_str + "*" + right_str;
                break;
            case OpType::Divide:
                f = left_str + " / " + right_str;
                break;
            case OpType::Power:
                f = left_str + "^" + right_str;
                break;
            default:
                break;
        }
        return f;
    }

    if (tree->type == NodeType::Function) {
        std::string arg_str;
        tree_to_string(tree->child, arg_str);

        switch (tree->func) {
            case FuncType::Sin: {
                f = "sin(" + arg_str + ")";
                break;
            }
            case FuncType::Sinh: {
                f = "sinh(" + arg_str + ")";
                break;
            }
            case FuncType::aSin: {
                f = "arcsin(" + arg_str + ")";
                break;
            }
            case FuncType::Cos: {
                f = "cos(" + arg_str + ")";
                break;
            }
            case FuncType::Cosh: {
                f = "cosh(" + arg_str + ")";
                break;
            }
            case FuncType::aCos: {
                f = "arccos(" + arg_str + ")";
                break;
            }
            case FuncType::Tan: {
                f = "tan(" + arg_str + ")";
                break;
            }
            case FuncType::Tanh: {
                f = "tanh(" + arg_str + ")";
                break;
            }
            case FuncType::aTan: {
                f = "arctan(" + arg_str + ")";
                break;
            }
            case FuncType::Log:
                f = "log(" + arg_str + ")";
                break;
            case FuncType::Ln:
                f = "ln(" + arg_str + ")";
                break;
            case FuncType::Exp:
                f = "exp(" + arg_str + ")";
                break;
            case FuncType::Sqrt:
                f = "sqrt(" + arg_str + ")";
                break;
            default:
                break;
        }
        return f;
    }

    return f;
}

void print_tree(const Tree &tree) {
    if (!tree) [[unlikely]] {
        std::cout << "Empty tree" << std::endl;
    }

    std::string f;
    tree_to_string(tree, f);
    std::cout << f << std::endl;
}
