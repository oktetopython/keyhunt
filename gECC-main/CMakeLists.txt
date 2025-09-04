cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libgecc LANGUAGES CUDA CXX)

include(GoogleTest)

enable_testing()
find_package(GTest REQUIRED)

# find_package(CUDA 11.1 REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80")
endif()
set(CMAKE_CUDA_STANDARD "17")

macro(cuda_target_compile_options TARGET VERBOSE)
  if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    if (${VERBOSE})
      target_compile_options(${TARGET} PRIVATE --ptxas-options=-v;-lineinfo;-ftz=true)
    endif ()
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  else()
    message(FATAL_ERROR "Unknown CMAKE_CUDA_COMPILER_ID ${CMAKE_CUDA_COMPILER_ID}")
  endif()
endmacro()

include_directories(include)

# Generate constants
file(GLOB CONSTANTS_GENERATER_SCRIPTS
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/constants_generator.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/constants.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/ccgen.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/ec_ops.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/ec.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/field.py"
)
set(GENERATED_CONSTANTS_HEADER
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_constants.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_ops_cc_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/ec_ops_add_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/ec_ops_dbl_details.h"
  "${CMAKE_CURRENT_BINARY_DIR}/include/fp_test_constants.h"
)
add_custom_command(OUTPUT ${GENERATED_CONSTANTS_HEADER}
  COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/scripts/constants_generator.py" --out "${CMAKE_CURRENT_BINARY_DIR}/include"
  DEPENDS ${CONSTANTS_GENERATER_SCRIPTS}
  COMMENT "Generating Constants..."
  VERBATIM
)
add_custom_target(generated_constants DEPENDS ${GENERATED_CONSTANTS_HEADER})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(test)
