#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <limits>
#include <map>
#include <set>
#include "apex_types.h"

namespace apex {

namespace exhaustive {

enum class VariableType { doubletype, longtype, stringtype } ;

struct VariableCheckpoint {
    size_t current_index = 0;
    size_t best_index = 0;
    size_t candidate_count = 0;
    std::string candidate_hash;
};

struct WindowCheckpoint {
    bool valid = false;
    double samples = 0.0;
    double accumulated = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

struct Checkpoint {
    bool valid = false;
    size_t k = 1;
    size_t kmax = 0;
    double cost = std::numeric_limits<double>::max();
    double best_cost = std::numeric_limits<double>::max();
    WindowCheckpoint window;
    std::vector<std::string> invalid_configs;
    std::map<std::string, VariableCheckpoint> variables;
};

inline void hash_append(uint64_t& hash, const std::string& value) {
    const uint64_t prime = 1099511628211ULL;
    for (char c : value) {
        hash ^= static_cast<unsigned char>(c);
        hash *= prime;
    }
}

inline std::string hash_to_string(uint64_t hash) {
    std::stringstream ss;
    ss << "fnv1a64:" << std::hex << hash;
    return ss.str();
}

class Variable {
public:
    std::vector<double> dvalues;
    std::vector<long> lvalues;
    std::vector<std::string> svalues;
    VariableType vtype;
    size_t current_index;
    size_t best_index;
    void * value; // for the client to get the values
    size_t max_index;
    Variable () = delete;
    Variable (VariableType vtype, void * ptr) : vtype(vtype), current_index(0),
        best_index(0), value(ptr), max_index(0) { }
    void set_current_value() {
        if (vtype == VariableType::doubletype) {
            *((double*)(value)) = dvalues[current_index];
        }
        else if (vtype == VariableType::longtype) {
            *((long*)(value)) = lvalues[current_index];
        }
        else {
            *((const char**)(value)) = svalues[current_index].c_str();
        }
    }
    size_t get_next_neighbor() {
        current_index++;
        if (current_index > max_index) {
            current_index = 0;
        }
        set_current_value();
        return current_index;
    }
    void save_best() { best_index = current_index; }
    void set_init() {
        max_index = (std::max(std::max(dvalues.size(),
            lvalues.size()), svalues.size())) - 1;
        current_index = max_index;
        set_current_value();
    }
    std::string getBest() {
        if (vtype == VariableType::doubletype) {
            *((double*)(value)) = dvalues[best_index];
            return std::to_string(dvalues[best_index]);
        }
        else if (vtype == VariableType::longtype) {
            *((long*)(value)) = lvalues[best_index];
            return std::to_string(lvalues[best_index]);
        }
        //else if (vtype == VariableType::stringtype) {
        *((const char**)(value)) = svalues[best_index].c_str();
        return svalues[best_index];
    }
    std::string toString() {
        if (vtype == VariableType::doubletype) {
            return std::to_string(dvalues[current_index]);
        }
        else if (vtype == VariableType::longtype) {
            return std::to_string(lvalues[current_index]);
        }
        //else if (vtype == VariableType::stringtype) {
        return svalues[current_index];
        //}
    }
    size_t candidate_count() const {
        if (vtype == VariableType::doubletype) {
            return dvalues.size();
        }
        if (vtype == VariableType::longtype) {
            return lvalues.size();
        }
        return svalues.size();
    }
    std::string candidate_hash() const {
        uint64_t hash = 14695981039346656037ULL;
        if (vtype == VariableType::doubletype) {
            hash_append(hash, "double:");
            for (double value : dvalues) {
                std::stringstream ss;
                ss << std::setprecision(
                    std::numeric_limits<double>::max_digits10) << value;
                hash_append(hash, ss.str());
                hash_append(hash, ";");
            }
        } else if (vtype == VariableType::longtype) {
            hash_append(hash, "long:");
            for (long value : lvalues) {
                hash_append(hash, std::to_string(value));
                hash_append(hash, ";");
            }
        } else {
            hash_append(hash, "string:");
            for (const std::string& value : svalues) {
                hash_append(hash, std::to_string(value.size()));
                hash_append(hash, ":");
                hash_append(hash, value);
                hash_append(hash, ";");
            }
        }
        return hash_to_string(hash);
    }
};

class Exhaustive {
private:
    double cost;
    double best_cost;
    size_t kmax;
    size_t k;
    std::map<std::string, Variable> vars;
    std::set<std::string> invalid_configs;
    WindowCheckpoint restored_window;
    //const size_t max_iterations{1000};
    //const size_t min_iterations{100};
    void advance_once() {
        for (auto& v : vars) {
            size_t index = v.second.get_next_neighbor();
            if (index != 0) {
                break;
            }
        }
    }
    size_t get_num_configurations() const {
        size_t count{1};
        for (const auto& v : vars) {
            count = count * v.second.candidate_count();
        }
        return count;
    }
    std::string current_config_key() const {
        std::stringstream key;
        for (const auto& v : vars) {
            key << v.first << "=" << v.second.current_index << ";";
        }
        return key.str();
    }
    bool current_config_invalid() const {
        return invalid_configs.count(current_config_key()) > 0;
    }
public:
    void evaluate(double new_cost);
    Exhaustive() :
        kmax(0), k(1) {
        cost = std::numeric_limits<double>::max();
        best_cost = cost;
        //std::cout << "New Session!" << std::endl;
    }
    bool converged() { return (k >= kmax); }
    void getNewSettings() {
        if (vars.empty()) { return; }
        size_t attempts{0};
        const size_t max_attempts = std::max<size_t>(1, get_num_configurations());
        do {
            advance_once();
            attempts++;
        } while (attempts < max_attempts && current_config_invalid());
        if (current_config_invalid() &&
            invalid_configs.size() >= max_attempts) {
            k = kmax;
        }
    }
    void saveBestSettings() {
        for (auto& v : vars) { v.second.getBest(); }
    }
    Checkpoint get_checkpoint() const {
        Checkpoint checkpoint;
        checkpoint.valid = true;
        checkpoint.k = k;
        checkpoint.kmax = kmax;
        checkpoint.cost = cost;
        checkpoint.best_cost = best_cost;
        checkpoint.window = restored_window;
        checkpoint.invalid_configs.assign(invalid_configs.begin(),
            invalid_configs.end());
        for (const auto& v : vars) {
            VariableCheckpoint variable;
            variable.current_index = v.second.current_index;
            variable.best_index = v.second.best_index;
            variable.candidate_count = v.second.candidate_count();
            variable.candidate_hash = v.second.candidate_hash();
            checkpoint.variables.insert(std::make_pair(v.first, variable));
        }
        return checkpoint;
    }
    bool restore_checkpoint(const Checkpoint& checkpoint) {
        if (!checkpoint.valid) { return false; }
        if (checkpoint.variables.size() != vars.size()) { return false; }
        if (checkpoint.kmax != 0 && checkpoint.kmax != kmax) { return false; }
        for (const auto& checkpoint_var : checkpoint.variables) {
            auto var = vars.find(checkpoint_var.first);
            if (var == vars.end()) { return false; }
            if (checkpoint_var.second.current_index > var->second.max_index ||
                checkpoint_var.second.best_index > var->second.max_index) {
                return false;
            }
            if (checkpoint_var.second.candidate_count != 0 &&
                checkpoint_var.second.candidate_count !=
                    var->second.candidate_count()) {
                return false;
            }
            if (!checkpoint_var.second.candidate_hash.empty() &&
                checkpoint_var.second.candidate_hash !=
                    var->second.candidate_hash()) {
                return false;
            }
        }
        k = checkpoint.k;
        if (k > kmax) { k = kmax; }
        if (k == 0) { k = 1; }
        cost = checkpoint.cost;
        best_cost = checkpoint.best_cost;
        invalid_configs.clear();
        invalid_configs.insert(checkpoint.invalid_configs.begin(),
            checkpoint.invalid_configs.end());
        restored_window = checkpoint.window;
        for (const auto& checkpoint_var : checkpoint.variables) {
            auto var = vars.find(checkpoint_var.first);
            var->second.current_index = checkpoint_var.second.current_index;
            var->second.best_index = checkpoint_var.second.best_index;
            var->second.set_current_value();
        }
        if (current_config_invalid()) {
            restored_window = WindowCheckpoint();
            getNewSettings();
        }
        return true;
    }
    bool get_restored_window(WindowCheckpoint& window) const {
        if (!restored_window.valid || restored_window.samples <= 0.0) {
            return false;
        }
        window = restored_window;
        return true;
    }
    bool consume_restored_window(WindowCheckpoint& window) {
        if (!get_restored_window(window)) {
            return false;
        }
        restored_window = WindowCheckpoint();
        return true;
    }
    void printBestSettings() {
        std::string d("[");
        for (auto v : vars) {
            std::cout << d << v.second.getBest();
            d = ",";
        }
        std::cout << "]" << std::endl;
    }
    size_t get_max_iterations();
    std::map<std::string, Variable>& get_vars() { return vars; }
    void add_var(std::string name, Variable var) {
        vars.insert(std::make_pair(name, var));
        kmax = get_max_iterations();
        /* get max iterations */
        //std::cout << "Max iterations : " << kmax << std::endl;
    }
};

} // exhaustive

} // apex
