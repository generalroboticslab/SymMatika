
#include "initialize_model.h"

// list RNG and variable list
std::random_device rd;
std::mt19937 gen(rd());
std::vector<std::string> var_list;

DataSet::DataSet(const std::string &filename, const bool &system_type, const std::vector<std::string> &variables) : name(filename), type(system_type), var_list(variables), pd_matrix_columns(0) {
    if (variables.size() < 2) {
        throw std::runtime_error("ERROR: Need at least one predictor variable and a target variable");
    }
    load_data();
}

// this model supports TXT and CSV files as input
void DataSet::load_data() {
    size_t period = name.find_last_of('.');
    if (period != std::string::npos) {
        std::string extension = name.substr(period + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        if (extension == "csv") {
            read_csv();
        } else if (extension == "txt") {
            read_txt();
        }
    }
}

void DataSet::read_csv() {
    std::ifstream file(name);
    if (!file.is_open()) {
        throw std::runtime_error("ERROR: Can't open file " + name);
    }

    std::string line;
    std::vector<std::vector<double>> rows;
    int num_vars = static_cast<int>(var_list.size());

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::vector<double> row_data;
        std::istringstream iss(line);

        bool has_comma = (line.find(',') != std::string::npos);
        if (has_comma) {
            std::string token;
            while (std::getline(iss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                try {
                    row_data.push_back(token.empty() ? 0.0 : std::stod(token));
                } catch (...) {
                    row_data.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            }
        } else {
            double value;
            while (iss >> value) {
                row_data.push_back(value);
            }
        }

        if (is_supervised()) {
            if (row_data.size() != static_cast<size_t>(num_vars)) {
                if (row_data.size() < static_cast<size_t>(num_vars))
                    row_data.resize(num_vars, std::numeric_limits<double>::quiet_NaN());
                else
                    row_data.resize(num_vars);
            }
        } else {
            size_t expected_cols = num_vars+2;
            if (row_data.size() < expected_cols) {
                throw std::runtime_error("ERROR: unequal expected columns and row data size");
            }
        }
        rows.push_back(std::move(row_data));
    }

    if (rows.empty()) {
        throw std::runtime_error("ERROR: No data found in CSV file");
    }

    if (is_supervised()) {
        Eigen::MatrixXd M(num_vars, rows.size());
        for (size_t i=0; i<rows.size(); i++) {
            for (int j=0; j<num_vars; j++) {
                M(j, i) = rows[i][j];
            }
        }
        data.clear();
        data.push_back(std::move(M));
        pd_matrix_columns = rows.size();
    } else {
        std::map<int, std::vector<std::vector<double>>> trial_data;
        for (auto &row : rows) {
            if (!row.empty()) {
                int trial = static_cast<int>(row[0]);
                trial_data[trial].push_back(row);
            }
        }

        size_t total_columns = 0;
        data.clear();
        data.reserve(trial_data.size());

        for (auto &[trial, trial_rows] : trial_data) {
            size_t T = trial_rows.size();
            Eigen::MatrixXd M(num_vars, T);
            total_columns += T;
            for (size_t t=0; t<T; t++) {
                for (int v=0; v<num_vars; v++) {
                    size_t data_index = v+2;
                    if (data_index < trial_rows[t].size()) {
                        M(v, t) = trial_rows[t][data_index];
                    } else {
                        throw std::runtime_error("ERROR: Can't read CSV value for variable " + std::to_string(v));
                    }
                }
            }
            data.push_back(std::move(M));
        }
        pd_matrix_columns = total_columns;
    }
}

void DataSet::read_txt() {
    std::ifstream file(name);
    if (!file.is_open()) {
        throw std::runtime_error("ERROR: Can't open file " + name);
    }

    std::string line;
    std::vector<std::vector<double>> rows;
    int num_vars = static_cast<int>(var_list.size());

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> row_data;
        std::istringstream iss(line);
        double value;
        while (iss >> value) {
            row_data.push_back(value);
        }
        if (!row_data.empty()) {
            if (is_supervised()) {
                if (row_data.size() != static_cast<size_t>(num_vars)) {
                    throw std::runtime_error("ERROR: TXT row size does not match variables size (row size=" + std::to_string(row_data.size()) + ", variables size=" + std::to_string(num_vars) + ")");
                }
            } else {
                size_t expected_cols = num_vars+2;
                if (row_data.size() < expected_cols) {
                    throw std::runtime_error("ERROR: unequal expected columns and row data size");
                }
            }
            rows.push_back(row_data);
        }
    }

    if (rows.empty()) {
        throw std::runtime_error("ERROR: No data found in TXT file");
    }

    if (is_supervised()) {
        Eigen::MatrixXd data_matrix(num_vars, rows.size());
        for (size_t i=0; i<rows.size(); i++) {
            for (int j=0; j<num_vars; j++) {
                data_matrix(j, i) = rows[i][j];
            }
        }
        data.clear();
        data.push_back(data_matrix);
        pd_matrix_columns = rows.size();
    } else {
        std::map<int, std::vector<std::vector<double>>> trial_data;
        for (const auto &row : rows) {
            if (!row.empty()) {
                int trial = static_cast<int>(row[0]);
                trial_data[trial].push_back(row);
            }
        }
        size_t total_columns = 0;
        data.clear();
        data.reserve(trial_data.size());
        for (auto i=trial_data.begin(); i!=trial_data.end(); i++) {
            const auto &trial_rows = i->second;
            size_t num_timepoints = trial_rows.size();
            Eigen::MatrixXd trial_matrix(num_vars, num_timepoints);
            total_columns += num_timepoints;
            for (size_t t=0; t<num_timepoints; t++) {
                for (int v=0; v<num_vars; v++) {
                    size_t data_index = v+2;
                    if (data_index < trial_rows[t].size()) {
                        trial_matrix(v, t) = trial_rows[t][data_index];
                    } else {
                        throw std::runtime_error("ERROR: Can't read TXT value for variable " + std::to_string(v));
                    }
                }
            }
            data.push_back(std::move(trial_matrix));
        }
        pd_matrix_columns = total_columns;
    }
}


const std::vector<std::string> &DataSet::get_variables() const {
    return var_list;
}

std::vector<std::string> DataSet::get_predictor_variables() const {
    if (!is_supervised()) {
        throw std::logic_error("ERROR: Cannot retrieve predictor variables from unsupervised dataset");
    }

    std::vector<std::string> predictors(var_list.begin(), var_list.end()-1);
    return predictors;
}

std::string DataSet::get_target_variable() const {
    if (!is_supervised()) {
        throw std::logic_error("ERROR: Cannot retrieve target variable from unsupervised dataset");
    }

    return var_list.back();
}

const std::vector<Eigen::MatrixXd> &DataSet::get_data() const {
    return data;
}

const Eigen::MatrixXd &DataSet::get_trial(size_t trial_index) const {
    if (trial_index >= data.size()) {
        throw std::out_of_range("ERROR: Trial index is out of range");
    }

    return data[trial_index];
}

size_t DataSet::get_trial_count() const {
    return data.size();
}

size_t DataSet::get_variable_count() const {
    return var_list.size();
}

bool DataSet::is_supervised() const {
    return type;
}

void initialize_variables(const DataSet& dataset) {
    if (dataset.is_supervised()) {
        var_list.assign(dataset.var_list.begin(), dataset.var_list.end()-1);
    } else {
        var_list.assign(dataset.var_list.begin(), dataset.var_list.end());
    }
}
