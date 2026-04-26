#include "apex.h"

#include <cstdlib>
#include <iostream>
#include <string>

int fail(const char* message) {
    std::cerr << message << std::endl;
    return EXIT_FAILURE;
}

int main(int argc, char** argv) {
    APEX_UNUSED(argc);
    APEX_UNUSED(argv);

    if (apex_kokkos_tuning_context_status("missing") !=
        APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("context status should be UNKNOWN without Kokkos");
    }
    if (apex_kokkos_tuning_context_status(nullptr) !=
        APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("null context status should be UNKNOWN without Kokkos");
    }
    if (apex_kokkos_tuning_context_converged("missing")) {
        return fail("context should not be converged without Kokkos");
    }

    if (apex_kokkos_tuning_context_variable_status(
            "kokkos.kernel_name", "missing") !=
        APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("context variable status should be UNKNOWN without Kokkos");
    }
    if (apex_kokkos_tuning_context_variable_status(nullptr, nullptr) !=
        APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("null context variable status should be UNKNOWN without Kokkos");
    }
    if (apex_kokkos_tuning_context_variable_converged(
            "kokkos.kernel_name", "missing")) {
        return fail("context variable should not be converged without Kokkos");
    }

    if (apex_kokkos_kernel_status("missing") !=
        APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("kernel status should be UNKNOWN without Kokkos");
    }
    if (apex_kokkos_kernel_status(nullptr) !=
        APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("null kernel status should be UNKNOWN without Kokkos");
    }
    if (apex_kokkos_kernel_converged("missing")) {
        return fail("kernel should not be converged without Kokkos");
    }

    std::string kernel_name = "owned_kernel_name";
    apex::kokkos::ReplayTarget target =
        apex::kokkos::ReplayTarget::kernel(kernel_name);
    kernel_name = "changed_after_target_construction";
    if (target.name != "owned_kernel_name") {
        return fail("ReplayTarget should own the kernel name string");
    }
    if (target.status() != APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("owned ReplayTarget kernel status should be UNKNOWN");
    }

    std::string variable_name = "kokkos.kernel_name";
    std::string variable_value = "owned_variable_value";
    apex::kokkos::ReplayTarget variable_target =
        apex::kokkos::ReplayTarget::context_variable(
            variable_name, variable_value);
    variable_name = "changed_variable_name";
    variable_value = "changed_variable_value";
    if (variable_target.name != "kokkos.kernel_name" ||
        variable_target.value != "owned_variable_value") {
        return fail("ReplayTarget should own context variable strings");
    }

    int prepare_calls = 0;
    int body_calls = 0;
    apex::kokkos::ReplayResult result =
        apex::kokkos::replay_kernel_until_converged(
            std::string("missing_kernel"), 3,
            [&] { prepare_calls++; },
            [&] { body_calls++; });

    if (result.initial_status != APEX_KOKKOS_TUNING_STATUS_UNKNOWN ||
        result.final_status != APEX_KOKKOS_TUNING_STATUS_UNKNOWN) {
        return fail("replay status should stay UNKNOWN without Kokkos");
    }
    if (result.converged || result.stopped_on_invalid) {
        return fail("replay should not converge or stop on invalid without Kokkos");
    }
    if (!result.max_replays_reached || result.replays != 3 ||
        prepare_calls != 3 || body_calls != 3) {
        return fail("replay should stop at max_replays without Kokkos");
    }

    return EXIT_SUCCESS;
}
