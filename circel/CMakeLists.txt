set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME Circel)

# These directories should be provided as arguments to `cmake`
set(LLVM_EXTERNAL_CIRCT_SOURCE_DIR CACHE STRING "")

# Include Directories
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories("${LLVM_EXTERNAL_CIRCT_SOURCE_DIR}/include")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)
# include_directories(${CMAKE_BINARY_DIR}/tools/circt/include)

# Subdirectories
add_subdirectory(lib)
add_subdirectory(test)
