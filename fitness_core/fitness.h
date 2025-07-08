
#ifndef FITNESS_H
#define FITNESS_H

#include "fitness_core/differentiation.h"

double fitness(const RCP<const Basic> &exp, const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &numeric_partials);

double supervised_fitness(const RCP<const Basic> &exp, const DataSet &dataset);

double unsupervised_fitness(const RCP<const Basic> &exp, const DataSet &dataset, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& numeric_partials);

#endif //FITNESS_H
