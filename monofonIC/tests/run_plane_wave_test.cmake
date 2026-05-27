# CMake script to run an analytical plane-wave regression test.
# Expected variables:
#   MONOFONIC_EXECUTABLE - built monofonIC binary
#   CONFIG_FILE          - config copied into the binary tree
#   OUTPUT_FILE          - expected main output file (removed before run)
#   DIAG_FILE            - plane-wave diagnostic HDF5 file
#   PYTHON_EXECUTABLE    - interpreter for the checker
#   CHECK_SCRIPT         - python script with analytical comparison logic

foreach(var MONOFONIC_EXECUTABLE CONFIG_FILE OUTPUT_FILE DIAG_FILE PYTHON_EXECUTABLE CHECK_SCRIPT)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not defined")
    endif()
endforeach()

file(REMOVE "${OUTPUT_FILE}" "${DIAG_FILE}")

message("Running monofonIC (plane-wave test) with config: ${CONFIG_FILE}")
execute_process(
    COMMAND "${MONOFONIC_EXECUTABLE}" "${CONFIG_FILE}"
    RESULT_VARIABLE MONOFONIC_RESULT
    OUTPUT_VARIABLE MONOFONIC_STDOUT
    ERROR_VARIABLE MONOFONIC_STDERR
    TIMEOUT 180
)

if(NOT MONOFONIC_RESULT EQUAL 0)
    message(FATAL_ERROR "monofonIC failed (exit ${MONOFONIC_RESULT})\n${MONOFONIC_STDOUT}\n${MONOFONIC_STDERR}")
endif()

if(NOT EXISTS "${DIAG_FILE}")
    message(FATAL_ERROR "Diagnostic file not produced: ${DIAG_FILE}")
endif()

message("Comparing diagnostics in ${DIAG_FILE} against analytical solution")
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "${CHECK_SCRIPT}"
            --diag "${DIAG_FILE}"
            --config "${CONFIG_FILE}"
    RESULT_VARIABLE CHECK_RESULT
    OUTPUT_VARIABLE CHECK_STDOUT
    ERROR_VARIABLE CHECK_STDERR
)

message("${CHECK_STDOUT}")
if(CHECK_STDERR)
    message("${CHECK_STDERR}")
endif()

if(NOT CHECK_RESULT EQUAL 0)
    message(FATAL_ERROR "Analytical comparison failed for ${CONFIG_FILE}")
endif()

message("Plane-wave analytical comparison passed.")
