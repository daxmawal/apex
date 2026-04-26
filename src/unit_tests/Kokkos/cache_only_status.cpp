#include "apex.h"
#include "Kokkos_Profiling_C_Interface.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
void kokkosp_init_library(int loadseq, uint64_t version,
    uint32_t ndevinfos, struct Kokkos_Profiling_KokkosPDeviceInfo* devinfos);
void kokkosp_finalize_library();
void kokkosp_declare_output_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info);
void kokkosp_declare_input_type(const char* name, const size_t id,
    Kokkos_Tools_VariableInfo& info);
void kokkosp_request_values(
    const size_t contextId,
    const size_t numContextVariables,
    const Kokkos_Tools_VariableValue* contextVariableValues,
    const size_t numTuningVariables,
    Kokkos_Tools_VariableValue* tuningVariableValues);
void kokkosp_begin_context(size_t contextId);
void kokkosp_end_context(const size_t contextId);
}

namespace {

constexpr const char* kTimerName = "cache_only_status_context";
constexpr const char* kInputName = "cache_only_status.input";
constexpr const char* kOutputName = "cache_only_status.output";

std::string context_key(int value) {
    std::stringstream ss;
    ss << "[cache_only_status.input:" << value
       << ",tree_node:cache_only_status_context]";
    return ss.str();
}

constexpr const char* kStatusCache = R"(Input_100:
  hash: 1001
  name: cache_only_status.input
  id: 100
  info.type: int64
  info.category: categorical
  info.valueQuantity: set
  info.candidates: [1,2,3,4]
Output_200:
  hash: 2001
  name: cache_only_status.output
  id: 200
  info.type: int64
  info.category: categorical
  info.valueQuantity: set
  info.candidates: [10,20,30,40]
Context_0:
  Name: "[cache_only_status.input:1,tree_node:cache_only_status_context]"
  Strategy: "exhaustive"
  Status: "converged"
  Converged: true
  Results:
    NumVars: 1
    id: 200
    value: 10
Context_1:
  Name: "[cache_only_status.input:2,tree_node:cache_only_status_context]"
  Strategy: "exhaustive"
  Status: "best_so_far"
  Converged: false
  BestSoFar: true
  Results:
    NumVars: 1
    id: 200
    value: 20
Context_2:
  Name: "[cache_only_status.input:3,tree_node:cache_only_status_context]"
  Strategy: "exhaustive"
  Status: "in_progress"
  Converged: false
  BestSoFar: true
  ExhaustiveState:
    Iteration: 2
    MaxIterations: 40
    Cost: 3.000000
    BestCost: 2.000000
    WindowSamples: 1
    WindowAccumulated: 3.000000
    WindowMinimum: 3.000000
    WindowMaximum: 3.000000
    NumVariables: 1
    Variable: "cache_only_status.output"
    CurrentIndex: 2
    BestIndex: 1
    CandidateCount: 4
    CandidateHash: "test-only"
    NumInvalidConfigs: 0
  Results:
    NumVars: 1
    id: 200
    value: 30
Context_3:
  Name: "[cache_only_status.input:4,tree_node:cache_only_status_context]"
  Strategy: "exhaustive"
  Status: "invalid"
  Converged: false
  Results:
    NumVars: 1
    id: 200
    value: 40
)";

struct IntSetInfo {
    explicit IntSetInfo(std::initializer_list<int64_t> values) {
        std::memset(&info, 0, sizeof(info));
        storage.assign(values.begin(), values.end());
        info.type = kokkos_value_int64;
        info.category = kokkos_value_categorical;
        info.valueQuantity = kokkos_value_set;
        info.candidates.set.size = storage.size();
        info.candidates.set.values.int_value = storage.data();
    }

    Kokkos_Tools_VariableInfo info;
    std::vector<int64_t> storage;
};

Kokkos_Tools_VariableValue make_int_variable_value(size_t id, int64_t value,
    Kokkos_Tools_VariableInfo* metadata) {
    Kokkos_Tools_VariableValue variable;
    std::memset(&variable, 0, sizeof(variable));
    variable.type_id = id;
    variable.value.int_value = value;
    variable.metadata = metadata;
    return variable;
}

std::string make_cache_path() {
    std::stringstream ss;
    ss << "/tmp/apex_cache_only_status_" << getpid() << ".yaml";
    return ss.str();
}

bool write_cache(const std::string& filename) {
    std::ofstream results(filename);
    results << kStatusCache;
    return results.good();
}

int fail(const std::string& message, const std::string& cache_file) {
    std::cerr << message << std::endl;
    unlink(cache_file.c_str());
    return 1;
}

void configure_apex(const std::string& cache_file) {
    setenv("APEX_KOKKOS_TUNING", "0", 1);
    setenv("APEX_KOKKOS_TUNING_CACHE_ONLY", "1", 1);
    setenv("APEX_KOKKOS_TUNING_CACHE_ALLOW_BEST_SO_FAR", "0", 1);
    setenv("APEX_KOKKOS_TUNING_CACHE", cache_file.c_str(), 1);
    setenv("APEX_SCREEN_OUTPUT", "0", 1);
    apex_set_use_kokkos_tuning(false);
    apex_set_use_kokkos_tuning_cache_only(true);
    apex_set_kokkos_tuning_cache_allow_best_so_far(false);
    apex_set_use_kokkos_verbose(false);
    apex_set_use_screen_output(false);
    apex_set_kokkos_tuning_cache(strdup(cache_file.c_str()));
}

int request_cached_value(size_t context_id, int input_value, int default_value,
    size_t input_id, Kokkos_Tools_VariableInfo* input_info,
    size_t output_id, Kokkos_Tools_VariableInfo* output_info) {
    std::array<Kokkos_Tools_VariableValue, 1> inputs{
        make_int_variable_value(input_id, input_value, input_info)
    };
    std::array<Kokkos_Tools_VariableValue, 1> outputs{
        make_int_variable_value(output_id, default_value, output_info)
    };
    kokkosp_begin_context(context_id);
    kokkosp_request_values(context_id, inputs.size(), inputs.data(),
        outputs.size(), outputs.data());
    kokkosp_end_context(context_id);
    return static_cast<int>(outputs[0].value.int_value);
}

int require_status(int value, apex_kokkos_tuning_status expected,
    const std::string& cache_file) {
    const std::string key = context_key(value);
    const apex_kokkos_tuning_status status =
        apex_kokkos_tuning_context_status(key.c_str());
    if (status != expected) {
        std::stringstream ss;
        ss << "Unexpected cache status for " << key << ": " << status;
        return fail(ss.str(), cache_file);
    }
    const bool expected_converged =
        expected == APEX_KOKKOS_TUNING_STATUS_CONVERGED;
    if (apex_kokkos_tuning_context_converged(key.c_str()) !=
        expected_converged) {
        return fail("Unexpected converged boolean for " + key, cache_file);
    }
    return 0;
}

} // namespace

int main() {
    const std::string cache_file = make_cache_path();
    if (!write_cache(cache_file)) {
        return fail("Could not write strict cache-only status cache.",
            cache_file);
    }

    configure_apex(cache_file);
    kokkosp_init_library(0, KOKKOSP_INTERFACE_VERSION, 0, nullptr);

    apex_profiler_handle profiler =
        apex_start(APEX_NAME_STRING, kTimerName);

    IntSetInfo input_info({1, 2, 3, 4});
    IntSetInfo output_info({10, 20, 30, 40});
    const size_t input_id = 11;
    const size_t output_id = 22;

    kokkosp_declare_input_type(kInputName, input_id, input_info.info);
    kokkosp_declare_output_type(kOutputName, output_id, output_info.info);

    int status = require_status(1, APEX_KOKKOS_TUNING_STATUS_CONVERGED,
        cache_file);
    if (status != 0) { return status; }
    status = require_status(2, APEX_KOKKOS_TUNING_STATUS_BEST_SO_FAR,
        cache_file);
    if (status != 0) { return status; }
    status = require_status(3, APEX_KOKKOS_TUNING_STATUS_IN_PROGRESS,
        cache_file);
    if (status != 0) { return status; }
    status = require_status(4, APEX_KOKKOS_TUNING_STATUS_INVALID,
        cache_file);
    if (status != 0) { return status; }

    if (request_cached_value(100, 1, -1, input_id, &input_info.info,
            output_id, &output_info.info) != 10) {
        return fail("Cache-only did not apply converged status.", cache_file);
    }
    if (request_cached_value(101, 2, -2, input_id, &input_info.info,
            output_id, &output_info.info) != -2) {
        return fail("Cache-only applied best-so-far without opt-in.",
            cache_file);
    }
    if (request_cached_value(102, 3, -3, input_id, &input_info.info,
            output_id, &output_info.info) != -3) {
        return fail("Cache-only applied in-progress without opt-in.",
            cache_file);
    }
    if (request_cached_value(103, 4, -4, input_id, &input_info.info,
            output_id, &output_info.info) != -4) {
        return fail("Cache-only applied invalid status.", cache_file);
    }

    apex_set_kokkos_tuning_cache_allow_best_so_far(true);
    setenv("APEX_KOKKOS_TUNING_CACHE_ALLOW_BEST_SO_FAR", "1", 1);

    if (request_cached_value(200, 2, -20, input_id, &input_info.info,
            output_id, &output_info.info) != 20) {
        return fail("Cache-only opt-in did not apply best-so-far.",
            cache_file);
    }
    if (request_cached_value(201, 3, -30, input_id, &input_info.info,
            output_id, &output_info.info) != -30) {
        return fail("Cache-only opt-in applied in-progress status.",
            cache_file);
    }
    if (request_cached_value(202, 4, -40, input_id, &input_info.info,
            output_id, &output_info.info) != -40) {
        return fail("Cache-only opt-in applied invalid status.", cache_file);
    }

    apex_stop(profiler);
    kokkosp_finalize_library();
    apex_finalize();
    apex_cleanup();

    unlink(cache_file.c_str());
    return 0;
}
