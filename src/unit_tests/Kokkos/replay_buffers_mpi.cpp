#include "apex.h"

#include <Kokkos_Core.hpp>
#include "Kokkos_Profiling_C_Interface.h"
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" {
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

constexpr const char* kKernelName = "replay_buffers_mpi_kernel";
constexpr int kSize = 256;

using ReplayExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using ReplayView = Kokkos::View<double*, ReplayExecutionSpace>;

int fail_mpi(const std::string& message, int rank) {
    std::cerr << "rank " << rank << ": " << message << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
}

double sentinel_for_rank(int rank) {
    return -41.0 - static_cast<double>(rank);
}

Kokkos_Tools_VariableInfo make_string_input_info() {
    Kokkos_Tools_VariableInfo info;
    std::memset(&info, 0, sizeof(info));
    info.type = kokkos_value_string;
    info.category = kokkos_value_categorical;
    info.valueQuantity = kokkos_value_unbounded;
    return info;
}

Kokkos_Tools_VariableInfo make_factor_output_info(
    std::vector<int64_t>& candidates) {
    Kokkos_Tools_VariableInfo info;
    std::memset(&info, 0, sizeof(info));
    info.type = kokkos_value_int64;
    info.category = kokkos_value_categorical;
    info.valueQuantity = kokkos_value_set;
    info.candidates.set.size = candidates.size();
    info.candidates.set.values.int_value = candidates.data();
    return info;
}

Kokkos_Tools_VariableValue make_kernel_input_value(size_t id,
    Kokkos_Tools_VariableInfo* info) {
    Kokkos_Tools_VariableValue value;
    std::memset(&value, 0, sizeof(value));
    value.type_id = id;
    value.metadata = info;
    std::strncpy(value.value.string_value, kKernelName,
        sizeof(value.value.string_value) - 1);
    return value;
}

Kokkos_Tools_VariableValue make_factor_output_value(size_t id,
    Kokkos_Tools_VariableInfo* info) {
    Kokkos_Tools_VariableValue value;
    std::memset(&value, 0, sizeof(value));
    value.type_id = id;
    value.metadata = info;
    value.value.int_value = 1;
    return value;
}

bool all_equal(const ReplayView& view, double expected) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    for (int i = 0 ; i < kSize ; i++) {
        if (std::abs(host(i) - expected) > 1.0e-12) {
            return false;
        }
    }
    return true;
}

bool matches_kernel_result(const ReplayView& a, const ReplayView& b,
    const ReplayView& c, int factor) {
    auto host_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    auto host_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    auto host_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c);
    for (int i = 0 ; i < kSize ; i++) {
        const double expected = host_a(i) + static_cast<double>(factor) *
            host_b(i);
        if (std::abs(host_c(i) - expected) > 1.0e-12) {
            return false;
        }
    }
    return true;
}

void run_kernel(ReplayView a, ReplayView b, ReplayView c, int factor) {
    ReplayView a_view = a;
    ReplayView b_view = b;
    ReplayView c_view = c;
    const int factor_value = factor;
    Kokkos::parallel_for(kKernelName,
        Kokkos::RangePolicy<ReplayExecutionSpace>(0, kSize),
        KOKKOS_LAMBDA(const int i) {
            c_view(i) = a_view(i) + static_cast<double>(factor_value) *
                b_view(i);
        });
}

} // namespace

int main(int argc, char* argv[]) {
    const char* configured_cache = std::getenv("APEX_KOKKOS_TUNING_CACHE");
    const std::string configured_cache_value = configured_cache != nullptr ?
        configured_cache : "";

    setenv("APEX_KOKKOS_TUNING", "1", 1);
    setenv("APEX_KOKKOS_TUNING_POLICY", "exhaustive", 1);
    setenv("APEX_KOKKOS_TUNING_WINDOW", "1", 1);
    setenv("APEX_SCREEN_OUTPUT", "0", 1);

    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        return fail_mpi("MPI replay test requires at least two ranks.", rank);
    }

    std::string cache_path;
    if (!configured_cache_value.empty()) {
        cache_path = configured_cache_value;
    } else {
        std::string generated_cache;
        if (rank == 0) {
            std::stringstream path;
            path << "/tmp/apex_kokkos_replay_buffers_mpi_" << getpid()
                 << ".yaml";
            generated_cache = path.str();
        }
        int cache_path_size = static_cast<int>(generated_cache.size());
        MPI_Bcast(&cache_path_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        cache_path.resize(static_cast<size_t>(cache_path_size));
        if (rank == 0) {
            std::memcpy(&cache_path[0], generated_cache.data(),
                static_cast<size_t>(cache_path_size));
        }
        MPI_Bcast(&cache_path[0], cache_path_size, MPI_CHAR, 0,
            MPI_COMM_WORLD);
    }
    setenv("APEX_KOKKOS_TUNING_CACHE", cache_path.c_str(), 1);
    if (rank == 0) {
        std::remove(cache_path.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Kokkos::initialize(argc, argv);
    {
        ReplayView a("a_mpi", kSize);
        ReplayView b("b_mpi", kSize);
        ReplayView c_real("c_real_mpi", kSize);
        ReplayView a_replay("a_replay_mpi", kSize);
        ReplayView b_replay("b_replay_mpi", kSize);
        ReplayView c_replay("c_replay_mpi", kSize);

        const double rank_offset = static_cast<double>(rank) * 1000.0;
        auto host_a = Kokkos::create_mirror_view(a);
        auto host_b = Kokkos::create_mirror_view(b);
        auto host_c_real = Kokkos::create_mirror_view(c_real);
        auto host_a_replay = Kokkos::create_mirror_view(a_replay);
        auto host_b_replay = Kokkos::create_mirror_view(b_replay);
        auto host_c_replay = Kokkos::create_mirror_view(c_replay);
        for (int i = 0 ; i < kSize ; i++) {
            host_a(i) = rank_offset + static_cast<double>(i);
            host_b(i) = static_cast<double>((i + rank) % 11) + 0.5;
            host_c_real(i) = sentinel_for_rank(rank);
            host_a_replay(i) = 0.0;
            host_b_replay(i) = 0.0;
            host_c_replay(i) = 0.0;
        }
        Kokkos::deep_copy(a, host_a);
        Kokkos::deep_copy(b, host_b);
        Kokkos::deep_copy(c_real, host_c_real);
        Kokkos::deep_copy(a_replay, host_a_replay);
        Kokkos::deep_copy(b_replay, host_b_replay);
        Kokkos::deep_copy(c_replay, host_c_replay);
        Kokkos::fence();

        std::vector<int64_t> factor_candidates{1, 2};
        Kokkos_Tools_VariableInfo kernel_info = make_string_input_info();
        Kokkos_Tools_VariableInfo factor_info =
            make_factor_output_info(factor_candidates);
        const size_t kernel_id = 701;
        const size_t factor_id = 801;
        kokkosp_declare_input_type("kokkos.kernel_name", kernel_id,
            kernel_info);
        kokkosp_declare_output_type("replay_buffers_mpi.factor", factor_id,
            factor_info);

        auto execute_tuned_context = [&](ReplayView in_a, ReplayView in_b,
            ReplayView out) -> int {
            Kokkos_Tools_VariableValue input =
                make_kernel_input_value(kernel_id, &kernel_info);
            Kokkos_Tools_VariableValue output =
                make_factor_output_value(factor_id, &factor_info);
            static size_t next_context = 7000;
            const size_t context = next_context++ +
                static_cast<size_t>(rank) * 100000;
            kokkosp_begin_context(context);
            kokkosp_request_values(context, 1, &input, 1, &output);
            const int factor = static_cast<int>(output.value.int_value);
            run_kernel(in_a, in_b, out, factor);
            Kokkos::fence();
            kokkosp_end_context(context);
            return factor;
        };

        size_t prepare_calls = 0;
        auto prepare = [&]() {
            prepare_calls++;
            Kokkos::deep_copy(a_replay, a);
            Kokkos::deep_copy(b_replay, b);
            Kokkos::deep_copy(c_replay, 0.0);
            Kokkos::fence();
        };
        auto replay_body = [&]() {
            execute_tuned_context(a_replay, b_replay, c_replay);
        };

        apex::kokkos::ReplayResult result =
            apex::kokkos::replay_kernel_until_converged(kKernelName, 32,
                prepare, replay_body);

        if (!result.converged) {
            return fail_mpi("MPI replay did not converge before max_replays.",
                rank);
        }
        if (result.stopped_on_invalid || result.max_replays_reached) {
            return fail_mpi("MPI replay stopped for the wrong reason.", rank);
        }
        if (prepare_calls != result.replays || result.replays == 0) {
            return fail_mpi("MPI replay did not call prepare once per replay.",
                rank);
        }
        if (!matches_kernel_result(a_replay, b_replay, c_replay, 1) &&
            !matches_kernel_result(a_replay, b_replay, c_replay, 2)) {
            return fail_mpi("MPI replay body did not update replay buffers.",
                rank);
        }
        if (!all_equal(c_real, sentinel_for_rank(rank))) {
            return fail_mpi("MPI replay modified the real output buffer.",
                rank);
        }

        const int real_factor = execute_tuned_context(a, b, c_real);
        if (!matches_kernel_result(a, b, c_real, real_factor)) {
            return fail_mpi("MPI final execution produced the wrong output.",
                rank);
        }
    }
    Kokkos::finalize();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
