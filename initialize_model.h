
#ifndef INITIALIZE_MODEL_H
#define INITIALIZE_MODEL_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_set>
#include <stdexcept>
#include <iomanip>
#include <stack>
#include <string>

#include <symengine/symbol.h>
#include <symengine/basic.h>
#include <symengine/expression.h>
#include <symengine/derivative.h>
#include <symengine/parser.h>
#include <symengine/real_double.h>
#include <symengine/functions.h>
#include <symengine/constants.h>
#include <symengine/subs.h>
#include <symengine/simplify.h>
#include <symengine/visitor.h>
#include <symengine/symengine_rcp.h>
#include <symengine/printers/latex.h>

#include <mutex>
#include <atomic>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include <sstream>
#include <fstream>

/* DataSet PARAMETERS:
 *
 * name: a string representation of the path to a .txt or .csv
 * file. For code use, we recommend placing the file in the project's
 * cmake-build-debug folder. For GUI use, simply upload the file in
 * the Data section. Instructions regarding valid data files must be
 * followed, and instructions are available on the GitHub
 *
 * type: a boolean variable for the system type. If searching for
 * EXPLICIT mappings, use TRUE. If searching for IMPLICIT relations,
 * use FALSE. We further explain these definitions in our video,
 * available on GitHub
 *
 * variables: a string vector, where the input variables are listed in
 * the order they appear in the dataset, and, if searching for explicit
 * mappings, a target variable is placed as the last element in the
 * vector
 */

struct DataSet {
    // parameters
    std::string name;
    bool type;
    std::vector<Eigen::MatrixXd> data;
    std::vector<std::string> var_list;
    size_t pd_matrix_columns;

    DataSet(const std::string &filename, const bool &system_type, const std::vector<std::string> &variables);

    void load_data();

    void read_txt();
    void read_csv();

    const std::vector<std::string> &get_variables() const;
    std::vector<std::string> get_predictor_variables() const;
    std::string get_target_variable() const;


    const std::vector<Eigen::MatrixXd> &get_data() const;
    const Eigen::MatrixXd &get_trial(size_t trial_index) const;

    size_t get_trial_count() const;
    size_t get_variable_count() const;
    bool is_supervised() const;
};


// RNG
extern std::random_device rd;
extern std::mt19937 gen;
extern std::vector<std::string> var_list;

// define parsimony bounds
const int PARSIMONY_SCALE = 30;

void initialize_variables(const DataSet &dataset);

#endif //INITIALIZE_MODEL_H
