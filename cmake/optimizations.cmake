set(PROJ_ROOT   "${ENGINE_ROOT}")
set(FITNESS_DIR "${PROJ_ROOT}/fitness_core")
set(TREE_DIR    "${PROJ_ROOT}/tree_construction")
set(EVOL_DIR    "${PROJ_ROOT}/evolution")

option(ENABLE_SANITIZER_ADDRESS   "Address sanitizer"           OFF)
option(ENABLE_SANITIZER_LEAK      "Leak sanitizer"              OFF)
option(ENABLE_SANITIZER_UNDEFINED "Undefined behavior sanitizer" OFF)

set(DATA_PROCESSING_SOURCES
        "${PROJ_ROOT}/initialize_model.cpp"
        "${PROJ_ROOT}/initialize_model.h"
)
add_library(data_processing STATIC ${DATA_PROCESSING_SOURCES})
target_include_directories(data_processing PUBLIC "${PROJ_ROOT}")
target_link_libraries(data_processing PUBLIC project_dependencies)

set(COMPUTATION_SOURCES
        "${TREE_DIR}/build_candidates.cpp"
        "${TREE_DIR}/build_candidates.h"
        "${TREE_DIR}/generate_population.cpp"
        "${TREE_DIR}/generate_population.h"
        "${FITNESS_DIR}/differentiation.cpp"
        "${FITNESS_DIR}/differentiation.h"
        "${FITNESS_DIR}/evaluate_tree.cpp"
        "${FITNESS_DIR}/evaluate_tree.h"
        "${FITNESS_DIR}/fitness.cpp"
        "${FITNESS_DIR}/fitness.h"
        "${EVOL_DIR}/genetic_algorithm.cpp"
        "${EVOL_DIR}/genetic_algorithm.h"
        "${EVOL_DIR}/genetic_operations.cpp"
        "${EVOL_DIR}/genetic_operations.h"
        "${EVOL_DIR}/motif_finder.cpp"
        "${EVOL_DIR}/motif_finder.h"
        "${PROJ_ROOT}/model.cpp"
        "${PROJ_ROOT}/model.h"
        "${PROJ_ROOT}/main.cpp"
)
add_library(computation STATIC ${COMPUTATION_SOURCES})
target_include_directories(computation PUBLIC
        "${FITNESS_DIR}"
        "${TREE_DIR}"
        "${EVOL_DIR}"
        "${PROJ_ROOT}"
)
target_link_libraries(computation PUBLIC project_dependencies)

set(SANITIZER_FLAGS       "")
set(SANITIZER_LINKER_FLAGS "")

if(ENABLE_SANITIZER_ADDRESS)
    list(APPEND SANITIZER_FLAGS        "-fsanitize=address" "-fno-omit-frame-pointer")
    list(APPEND SANITIZER_LINKER_FLAGS "-fsanitize=address")
endif()
if(ENABLE_SANITIZER_LEAK)
    list(APPEND SANITIZER_FLAGS        "-fsanitize=leak" "-fno-omit-frame-pointer")
    list(APPEND SANITIZER_LINKER_FLAGS "-fsanitize=leak")
endif()
if(ENABLE_SANITIZER_UNDEFINED)
    list(APPEND SANITIZER_FLAGS        "-fsanitize=undefined")
    list(APPEND SANITIZER_LINKER_FLAGS "-fsanitize=undefined")
endif()

target_compile_options(data_processing
        PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -O2 -ftree-vectorize -funroll-loops -fstrict-aliasing -march=native
        ${SANITIZER_FLAGS}
        >
)
target_compile_options(computation
        PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -O3 -march=native -funroll-loops -ftree-vectorize -fopenmp
        ${SANITIZER_FLAGS}
        >
)
if(SANITIZER_LINKER_FLAGS)
    target_link_options(data_processing PRIVATE ${SANITIZER_LINKER_FLAGS})
    target_link_options(computation PRIVATE ${SANITIZER_LINKER_FLAGS})
endif()
