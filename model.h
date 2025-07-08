
#ifndef MODEL_H
#define MODEL_H

#include "evolution/genetic_algorithm.h"

// define model structure and core run function
struct Model {
private:
    DataSet dataset;
    const int initial_pop;
    const int max_depth;
    std::vector<bool> allowed_ops;

    std::vector<FFPair> final_candidates;
    size_t gen_count;
    double avg_loss;
    std::vector<std::pair<int,double>> loss_history;
    std::function<void(int,double)> progress;
    std::atomic<bool> stop_requested {false};

    static int adjust_depth(int d) {
        d = std::clamp(d, 1, 5);
        static const int depth_map[6] = {0, 2, 3, 4, 5, 6};
        return depth_map[d];
    }

public:

    /* MODEL PARAMETERS:
     *
     * dataset: .txt or .csv file with columns corresponding to variables
     * at certain samples. Further instructions on valid datasets can be found
     * in GitHub documentation
     *
     * initial_pop: initial population size of each island, by default 10000
     *
     * max_depth: size of mathematical expressions -- range is: [1, 2, 3, 5, 5],
     * where 1 = very small, 2 = small, 3 = medium, 4 = large, 5 = very large
     *
     * allowed_ops: boolean vector of size 18, where each element determines whether
     * a binary or unary operator is in use. The operators in order are:
     * ADD, SUBTRACT, MULTIPLY, DIVIDE, POWER, SIN, SINH, ASIN, COS, COSH, ACOS, TAN,
     * TANH, ATAN, LOG, LN, EXP, SQRT
     * Each element of allowed_ops corresponds to enabling a certain operator
     */

    Model(const DataSet &dataset_) : dataset(dataset_), initial_pop(10000), max_depth(5), allowed_ops(allowed_starting_weights) {}

    Model(const DataSet &dataset_, int pop_size_) : dataset(dataset_), initial_pop(pop_size_), max_depth(5), allowed_ops(allowed_starting_weights) {}

    Model(const DataSet &dataset_, int pop_size_, int max_depth_) : dataset(dataset_), initial_pop(pop_size_), max_depth(adjust_depth(max_depth_)), allowed_ops(allowed_starting_weights) {}

    Model(const DataSet &dataset_, int pop_size_, int max_depth_, std::vector<bool> allowed_) : dataset(dataset_), initial_pop(pop_size_), max_depth(adjust_depth(max_depth_)), allowed_ops(allowed_) {}

    void run();

    void request_stop() {
        stop_requested = true;
    }

    void set_progress(std::function<void(size_t,double)> p);

    const std::vector<std::pair<int,double>> &get_loss_history() const {
        return loss_history;
    }

    std::vector<std::pair<std::string,double>> return_final_candidates() const;
};

void easy_run();

#endif //MODEL_H
