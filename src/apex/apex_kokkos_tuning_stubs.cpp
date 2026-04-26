/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "apex.h"

#ifndef APEX_WITH_KOKKOS

extern "C" {

APEX_EXPORT apex_kokkos_tuning_status apex_kokkos_tuning_context_status(
    const char*) {
    return APEX_KOKKOS_TUNING_STATUS_UNKNOWN;
}

APEX_EXPORT bool apex_kokkos_tuning_context_converged(const char*) {
    return false;
}

APEX_EXPORT apex_kokkos_tuning_status
apex_kokkos_tuning_context_variable_status(const char*, const char*) {
    return APEX_KOKKOS_TUNING_STATUS_UNKNOWN;
}

APEX_EXPORT bool apex_kokkos_tuning_context_variable_converged(
    const char*, const char*) {
    return false;
}

APEX_EXPORT apex_kokkos_tuning_status apex_kokkos_kernel_status(
    const char*) {
    return APEX_KOKKOS_TUNING_STATUS_UNKNOWN;
}

APEX_EXPORT bool apex_kokkos_kernel_converged(const char*) {
    return false;
}

} // extern "C"

#endif
