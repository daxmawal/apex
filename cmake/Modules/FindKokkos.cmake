# - Try to find LibKokkos
# Once done this will define
#  Kokkos_FOUND - System has Kokkos
#  Kokkos_INCLUDE_DIRS - The Kokkos include directories
#  Kokkos_LIBRARIES - The libraries needed to use Kokkos
#  Kokkos_DEFINITIONS - Compiler switches required for using Kokkos

if(NOT DEFINED Kokkos_ROOT)
	if(DEFINED ENV{Kokkos_ROOT})
		# message("   env Kokkos_ROOT is defined as $ENV{Kokkos_ROOT}")
		set(Kokkos_ROOT $ENV{Kokkos_ROOT})
	endif()
endif()

message("Kokkos_ROOT is defined as ${Kokkos_ROOT}")

if(Kokkos_ROOT AND Kokkos_DIR)
	file(TO_CMAKE_PATH "${Kokkos_ROOT}" _APEX_KOKKOS_ROOT_PATH)
	file(TO_CMAKE_PATH "${Kokkos_DIR}" _APEX_KOKKOS_DIR_PATH)
	string(FIND "${_APEX_KOKKOS_DIR_PATH}" "${_APEX_KOKKOS_ROOT_PATH}" _APEX_KOKKOS_DIR_IN_ROOT)
	if(NOT _APEX_KOKKOS_DIR_IN_ROOT EQUAL 0)
		unset(Kokkos_DIR CACHE)
		unset(Kokkos_DIR)
	endif()
endif()

if(NOT DEFINED CUDAToolkit_ROOT)
	if(DEFINED ENV{CUDA_ROOT})
		set(CUDAToolkit_ROOT $ENV{CUDA_ROOT} CACHE PATH "CUDA Toolkit root")
	elseif(DEFINED ENV{CUDA_PATH})
		set(CUDAToolkit_ROOT $ENV{CUDA_PATH} CACHE PATH "CUDA Toolkit root")
	endif()
endif()

if(Kokkos_ROOT)
	find_package(Kokkos CONFIG QUIET
		PATHS
			${Kokkos_ROOT}
			${Kokkos_ROOT}/lib/cmake/Kokkos
			${Kokkos_ROOT}/lib64/cmake/Kokkos
		NO_DEFAULT_PATH)
else()
	find_package(Kokkos CONFIG QUIET)
endif()

if(Kokkos_FOUND AND TARGET Kokkos::kokkos)
	get_target_property(Kokkos_INCLUDE_DIRS Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
	if(NOT Kokkos_INCLUDE_DIRS)
		get_target_property(Kokkos_INCLUDE_DIRS Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
	endif()
	set(_APEX_KOKKOS_INCLUDE_DIRS)
	foreach(_APEX_KOKKOS_INCLUDE_DIR IN LISTS Kokkos_INCLUDE_DIRS)
		list(APPEND _APEX_KOKKOS_INCLUDE_DIRS ${_APEX_KOKKOS_INCLUDE_DIR})
		if(EXISTS "${_APEX_KOKKOS_INCLUDE_DIR}/impl/Kokkos_Profiling_C_Interface.h")
			list(APPEND _APEX_KOKKOS_INCLUDE_DIRS "${_APEX_KOKKOS_INCLUDE_DIR}/impl")
		endif()
	endforeach()
	list(REMOVE_DUPLICATES _APEX_KOKKOS_INCLUDE_DIRS)
	set(Kokkos_INCLUDE_DIRS ${_APEX_KOKKOS_INCLUDE_DIRS})
	list(GET Kokkos_INCLUDE_DIRS 0 Kokkos_INCLUDE_DIR)
	set(Kokkos_LIBRARY Kokkos::kokkos)
	set(Kokkos_LIBRARIES Kokkos::kokkos)
	add_definitions(-DAPEX_HAVE_KOKKOS)
	return()
endif()

find_path(Kokkos_INCLUDE_DIR NAMES Kokkos_Core.hpp
	HINTS ${Kokkos_ROOT}/include $ENV{Kokkos_ROOT}/include)
find_library(Kokkos_LIBRARY NAMES kokkoscore
        HINTS ${Kokkos_ROOT}/* $ENV{Kokkos_ROOT}/*)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set Kokkos_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Kokkos  DEFAULT_MSG
                                  Kokkos_LIBRARY Kokkos_INCLUDE_DIR)

mark_as_advanced(Kokkos_INCLUDE_DIR Kokkos_LIBRARY)

if(Kokkos_FOUND)
  set(Kokkos_INCLUDE_DIRS ${Kokkos_INCLUDE_DIR})
  set(Kokkos_LIBRARIES ${Kokkos_LIBRARY})
  set(Kokkos_DIR ${Kokkos_ROOT})
  add_definitions(-DAPEX_HAVE_KOKKOS)
endif()
