#include "apex.h"
#include "Kokkos_Profiling_C_Interface.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <limits.h>

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

constexpr const char* kTimerName = "restart_replay_context";
constexpr const char* kContextKey =
    "[restart_replay.input:7,tree_node:restart_replay_context]";

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

struct Variables {
    size_t input_id = 101;
    size_t output_a_id = 201;
    size_t output_b_id = 202;
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

std::string slurp(const std::string& path) {
    std::ifstream input(path);
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

int fail(const std::string& message) {
    std::cerr << message << std::endl;
    return 1;
}

bool write_file(const std::string& path, const std::string& contents) {
    std::ofstream output(path);
    output << contents;
    return output.good();
}

bool write_cache_with_invalid_config(const std::string& path,
    std::string contents, const std::string& invalid_config) {
    const std::string marker = "    NumInvalidConfigs: 0\n";
    const size_t offset = contents.find(marker);
    if (offset == std::string::npos) {
        return false;
    }
    std::stringstream replacement;
    replacement << "    NumInvalidConfigs: 1\n"
        << "    InvalidConfig: \"" << invalid_config << "\"\n";
    contents.replace(offset, marker.size(), replacement.str());
    return write_file(path, contents);
}

void configure_apex(const std::string& cache_file, bool cache_only,
    int window = 1) {
    const std::string window_text = std::to_string(window);
    setenv("APEX_KOKKOS_TUNING", cache_only ? "0" : "1", 1);
    setenv("APEX_KOKKOS_TUNING_CACHE_ONLY", cache_only ? "1" : "0", 1);
    setenv("APEX_KOKKOS_TUNING_CACHE_ALLOW_BEST_SO_FAR", "0", 1);
    setenv("APEX_KOKKOS_TUNING_CACHE", cache_file.c_str(), 1);
    setenv("APEX_KOKKOS_TUNING_POLICY", "exhaustive", 1);
    setenv("APEX_KOKKOS_TUNING_WINDOW", window_text.c_str(), 1);
    setenv("APEX_SCREEN_OUTPUT", "0", 1);
    apex_set_use_kokkos_tuning(!cache_only);
    apex_set_use_kokkos_tuning_cache_only(cache_only);
    apex_set_kokkos_tuning_cache_allow_best_so_far(false);
    apex_set_use_kokkos_verbose(false);
    apex_set_use_screen_output(false);
    apex_set_kokkos_tuning_window(window);
    apex_set_kokkos_tuning_policy(strdup("exhaustive"));
    apex_set_kokkos_tuning_cache(strdup(cache_file.c_str()));
}

Variables declare_variables(IntSetInfo& input_info, IntSetInfo& output_a_info,
    IntSetInfo& output_b_info) {
    Variables vars;
    kokkosp_declare_input_type("restart_replay.input", vars.input_id,
        input_info.info);
    kokkosp_declare_output_type("restart_replay.output_a", vars.output_a_id,
        output_a_info.info);
    kokkosp_declare_output_type("restart_replay.output_b", vars.output_b_id,
        output_b_info.info);
    return vars;
}

int run_tuning_child(const std::string& cache_file, const std::string& log_file,
    int iterations, bool incompatible_space = false, int window = 1) {
    configure_apex(cache_file, false, window);
    kokkosp_init_library(0, KOKKOSP_INTERFACE_VERSION, 0, nullptr);

    apex_profiler_handle profiler =
        apex_start(APEX_NAME_STRING, kTimerName);

    IntSetInfo input_info({7});
    IntSetInfo output_a_info({0, 1, 2});
    IntSetInfo output_b_info(
        incompatible_space ? std::initializer_list<int64_t>{0, 1, 2, 3, 4} :
            std::initializer_list<int64_t>{0, 1, 2, 3});
    Variables vars = declare_variables(input_info, output_a_info,
        output_b_info);

    std::ofstream log(log_file);
    if (!log.good()) {
        return fail("Could not open restart replay log: " + log_file);
    }

    for (int i = 0 ; i < iterations ; i++) {
        std::array<Kokkos_Tools_VariableValue, 1> inputs{
            make_int_variable_value(vars.input_id, 7, &input_info.info)
        };
        std::array<Kokkos_Tools_VariableValue, 2> outputs{
            make_int_variable_value(vars.output_a_id, -1,
                &output_a_info.info),
            make_int_variable_value(vars.output_b_id, -1,
                &output_b_info.info)
        };
        size_t context_id = 1000 + i;
        kokkosp_begin_context(context_id);
        kokkosp_request_values(context_id, inputs.size(), inputs.data(),
            outputs.size(), outputs.data());
        log << outputs[0].value.int_value << ","
            << outputs[1].value.int_value << std::endl;
        usleep(1000);
        kokkosp_end_context(context_id);
    }

    if (apex_kokkos_tuning_context_converged(kContextKey)) {
        return fail("Partial exhaustive context was reported as converged.");
    }
    if (apex_kokkos_tuning_context_status(kContextKey) !=
        APEX_KOKKOS_TUNING_STATUS_IN_PROGRESS) {
        return fail("Partial exhaustive context did not report in-progress status.");
    }

    apex_stop(profiler);
    kokkosp_finalize_library();
    apex_finalize();
    return 0;
}

int run_cache_only_child(const std::string& cache_file) {
    configure_apex(cache_file, true);
    kokkosp_init_library(0, KOKKOSP_INTERFACE_VERSION, 0, nullptr);

    apex_profiler_handle profiler =
        apex_start(APEX_NAME_STRING, kTimerName);

    IntSetInfo input_info({7});
    IntSetInfo output_a_info({0, 1, 2});
    IntSetInfo output_b_info({0, 1, 2, 3});
    Variables vars = declare_variables(input_info, output_a_info,
        output_b_info);

    std::array<Kokkos_Tools_VariableValue, 1> inputs{
        make_int_variable_value(vars.input_id, 7, &input_info.info)
    };
    std::array<Kokkos_Tools_VariableValue, 2> outputs{
        make_int_variable_value(vars.output_a_id, -17, &output_a_info.info),
        make_int_variable_value(vars.output_b_id, -19, &output_b_info.info)
    };

    kokkosp_request_values(2000, inputs.size(), inputs.data(),
        outputs.size(), outputs.data());

    if (outputs[0].value.int_value != -17 ||
        outputs[1].value.int_value != -19) {
        std::stringstream ss;
        ss << "Cache-only replay unexpectedly applied a partial best-so-far context: "
           << outputs[0].value.int_value << ","
           << outputs[1].value.int_value;
        return fail(ss.str());
    }
    if (apex_kokkos_tuning_context_converged(kContextKey)) {
        return fail("Cached best-so-far context was reported as converged.");
    }
    if (apex_kokkos_tuning_context_status(kContextKey) !=
        APEX_KOKKOS_TUNING_STATUS_IN_PROGRESS) {
        return fail("Cached partial context did not report in-progress status.");
    }

    apex_stop(profiler);
    kokkosp_finalize_library();
    apex_finalize();
    return 0;
}

int run_wrapper_child(const std::string& cache_file,
    const std::string& log_file) {
    configure_apex(cache_file, false);
    kokkosp_init_library(0, KOKKOSP_INTERFACE_VERSION, 0, nullptr);

    apex_profiler_handle profiler =
        apex_start(APEX_NAME_STRING, kTimerName);

    IntSetInfo input_info({7});
    IntSetInfo output_a_info({0, 1, 2});
    IntSetInfo output_b_info({0, 1, 2, 3});
    Variables vars = declare_variables(input_info, output_a_info,
        output_b_info);

    std::ofstream log(log_file);
    if (!log.good()) {
        return fail("Could not open replay wrapper log: " + log_file);
    }

    size_t prepare_calls = 0;
    size_t body_calls = 0;
    auto prepare = [&]() {
        prepare_calls++;
    };
    auto body = [&]() {
        std::array<Kokkos_Tools_VariableValue, 1> inputs{
            make_int_variable_value(vars.input_id, 7, &input_info.info)
        };
        std::array<Kokkos_Tools_VariableValue, 2> outputs{
            make_int_variable_value(vars.output_a_id, -1,
                &output_a_info.info),
            make_int_variable_value(vars.output_b_id, -1,
                &output_b_info.info)
        };
        size_t context_id = 3000 + body_calls;
        body_calls++;
        kokkosp_begin_context(context_id);
        kokkosp_request_values(context_id, inputs.size(), inputs.data(),
            outputs.size(), outputs.data());
        log << outputs[0].value.int_value << ","
            << outputs[1].value.int_value << std::endl;
        usleep(1000);
        kokkosp_end_context(context_id);
    };

    apex::kokkos::ReplayResult result =
        apex::kokkos::replay_context_until_converged(kContextKey, 20,
            prepare, body);

    if (!result.converged ||
        result.final_status != APEX_KOKKOS_TUNING_STATUS_CONVERGED) {
        return fail("Replay wrapper did not stop on convergence.");
    }
    if (result.max_replays_reached || result.stopped_on_invalid) {
        return fail("Replay wrapper stopped for the wrong reason.");
    }
    if (result.replays != 12) {
        std::stringstream ss;
        ss << "Replay wrapper ran an unexpected number of iterations: "
           << result.replays;
        return fail(ss.str());
    }
    if (prepare_calls != result.replays || body_calls != result.replays) {
        return fail("Replay wrapper did not call prepare/body once per replay.");
    }

    apex_stop(profiler);
    kokkosp_finalize_library();
    apex_finalize();
    return 0;
}

std::string executable_path(const char* argv0) {
    char buffer[PATH_MAX];
    ssize_t size = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (size > 0) {
        buffer[size] = '\0';
        return std::string(buffer);
    }
    return std::string(argv0);
}

int run_child(const std::string& executable, const std::string& mode,
    const std::string& cache_file, const std::string& log_file = "") {
    pid_t pid = fork();
    if (pid < 0) {
        return fail("fork failed: " + std::string(std::strerror(errno)));
    }
    if (pid == 0) {
        std::vector<char*> args;
        args.push_back(const_cast<char*>(executable.c_str()));
        args.push_back(const_cast<char*>(mode.c_str()));
        args.push_back(const_cast<char*>(cache_file.c_str()));
        if (!log_file.empty()) {
            args.push_back(const_cast<char*>(log_file.c_str()));
        }
        args.push_back(nullptr);
        execv(executable.c_str(), args.data());
        std::cerr << "execv failed: " << std::strerror(errno) << std::endl;
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return fail("waitpid failed: " + std::string(std::strerror(errno)));
    }
    if (!WIFEXITED(status)) {
        return fail("Child process did not exit normally for mode " + mode);
    }
    return WEXITSTATUS(status);
}

std::set<std::string> read_combinations(const std::string& log_file) {
    std::set<std::string> combinations;
    std::ifstream input(log_file);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            combinations.insert(line);
        }
    }
    return combinations;
}

int run_driver(const char* argv0) {
    std::stringstream base;
    base << "/tmp/apex_kokkos_restart_replay_" << getpid();
    const std::string cache_file = base.str() + ".yaml";
    const std::string incompatible_cache_file =
        base.str() + ".incompatible.yaml";
    const std::string invalid_cache_file =
        base.str() + ".invalid.yaml";
    const std::string window_cache_file =
        base.str() + ".window.yaml";
    const std::string wrapper_cache_file =
        base.str() + ".wrapper.yaml";
    const std::string run1_log = base.str() + ".run1";
    const std::string run2_log = base.str() + ".run2";
    const std::string incompatible_log = base.str() + ".incompatible";
    const std::string invalid_log = base.str() + ".invalid";
    const std::string window_run1_log = base.str() + ".window1";
    const std::string window_run2_log = base.str() + ".window2";
    const std::string wrapper_log = base.str() + ".wrapper";
    const std::string executable = executable_path(argv0);

    unlink(cache_file.c_str());
    unlink(incompatible_cache_file.c_str());
    unlink(invalid_cache_file.c_str());
    unlink(window_cache_file.c_str());
    unlink(wrapper_cache_file.c_str());
    unlink(run1_log.c_str());
    unlink(run2_log.c_str());
    unlink(incompatible_log.c_str());
    unlink(invalid_log.c_str());
    unlink(window_run1_log.c_str());
    unlink(window_run2_log.c_str());
    unlink(wrapper_log.c_str());

    int status = run_child(executable, "run1", cache_file, run1_log);
    if (status != 0) { return status; }

    const std::string run1_cache = slurp(cache_file);
    if (run1_cache.find("Status: \"in_progress\"") == std::string::npos ||
        run1_cache.find("Converged: false") == std::string::npos ||
        run1_cache.find("BestSoFar: true") == std::string::npos ||
        run1_cache.find("ExhaustiveState:") == std::string::npos ||
        run1_cache.find("MaxIterations:") == std::string::npos ||
        run1_cache.find("CandidateCount:") == std::string::npos ||
        run1_cache.find("CandidateHash:") == std::string::npos ||
        run1_cache.find("WindowSamples:") == std::string::npos ||
        run1_cache.find("NumInvalidConfigs:") == std::string::npos) {
        return fail("Run 1 did not write a partial exhaustive checkpoint.");
    }

    status = run_child(executable, "cache-only", cache_file);
    if (status != 0) { return status; }

    if (!write_file(incompatible_cache_file, run1_cache)) {
        return fail("Could not create incompatible cache copy.");
    }
    status = run_child(executable, "run-incompatible",
        incompatible_cache_file, incompatible_log);
    if (status != 0) { return status; }
    const std::string incompatible_run = slurp(incompatible_log);
    if (incompatible_run.compare(0, 4, "0,0\n") != 0) {
        return fail("Incompatible exhaustive checkpoint was restored instead "
            "of restarting from the beginning.");
    }

    if (!write_cache_with_invalid_config(invalid_cache_file, run1_cache,
            "restart_replay.output_a=2;restart_replay.output_b=1;")) {
        return fail("Could not create invalid-config cache copy.");
    }
    status = run_child(executable, "run-invalid", invalid_cache_file,
        invalid_log);
    if (status != 0) { return status; }
    const std::string invalid_run = slurp(invalid_log);
    if (invalid_run.compare(0, 4, "0,2\n") != 0) {
        return fail("Invalid exhaustive checkpoint configuration was not skipped.");
    }

    status = run_child(executable, "run2", cache_file, run2_log);
    if (status != 0) { return status; }

    std::set<std::string> run1 = read_combinations(run1_log);
    std::set<std::string> run2 = read_combinations(run2_log);
    for (const auto& combination : run2) {
        if (run1.count(combination) > 0) {
            return fail("Run 2 repeated an exhaustive combination from run 1: "
                + combination);
        }
    }

    status = run_child(executable, "run-window1", window_cache_file,
        window_run1_log);
    if (status != 0) { return status; }
    const std::string window_cache = slurp(window_cache_file);
    if (window_cache.find("WindowSamples: 2") == std::string::npos) {
        return fail("Partial intra-configuration window was not checkpointed.");
    }
    status = run_child(executable, "run-window2", window_cache_file,
        window_run2_log);
    if (status != 0) { return status; }
    const std::string window_run2 = slurp(window_run2_log);
    if (window_run2 != "0,0\n1,0\n") {
        return fail("Partial intra-configuration window was not resumed.");
    }

    status = run_child(executable, "run-wrapper", wrapper_cache_file,
        wrapper_log);
    if (status != 0) { return status; }

    unlink(cache_file.c_str());
    unlink(incompatible_cache_file.c_str());
    unlink(invalid_cache_file.c_str());
    unlink(window_cache_file.c_str());
    unlink(wrapper_cache_file.c_str());
    unlink(run1_log.c_str());
    unlink(run2_log.c_str());
    unlink(incompatible_log.c_str());
    unlink(invalid_log.c_str());
    unlink(window_run1_log.c_str());
    unlink(window_run2_log.c_str());
    unlink(wrapper_log.c_str());
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc > 1) {
        const std::string mode = argv[1];
        if (mode == "run1") {
            if (argc != 4) { return fail("run1 requires cache and log paths"); }
            return run_tuning_child(argv[2], argv[3], 5);
        }
        if (mode == "run2") {
            if (argc != 4) { return fail("run2 requires cache and log paths"); }
            return run_tuning_child(argv[2], argv[3], 4);
        }
        if (mode == "run-incompatible") {
            if (argc != 4) {
                return fail("run-incompatible requires cache and log paths");
            }
            return run_tuning_child(argv[2], argv[3], 1, true);
        }
        if (mode == "run-invalid") {
            if (argc != 4) {
                return fail("run-invalid requires cache and log paths");
            }
            return run_tuning_child(argv[2], argv[3], 1);
        }
        if (mode == "run-window1") {
            if (argc != 4) {
                return fail("run-window1 requires cache and log paths");
            }
            return run_tuning_child(argv[2], argv[3], 2, false, 3);
        }
        if (mode == "run-window2") {
            if (argc != 4) {
                return fail("run-window2 requires cache and log paths");
            }
            return run_tuning_child(argv[2], argv[3], 2, false, 3);
        }
        if (mode == "run-wrapper") {
            if (argc != 4) {
                return fail("run-wrapper requires cache and log paths");
            }
            return run_wrapper_child(argv[2], argv[3]);
        }
        if (mode == "cache-only") {
            if (argc != 3) { return fail("cache-only requires a cache path"); }
            return run_cache_only_child(argv[2]);
        }
        return fail("Unknown restart replay mode: " + mode);
    }
    return run_driver(argv[0]);
}
