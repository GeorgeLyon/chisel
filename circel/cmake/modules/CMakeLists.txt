# Generate a list of CMake library targets so that other CMake projects can
# link against them.
set(CIRCEL_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/circel)
set(circel_cmake_builddir "${CMAKE_BINARY_DIR}/${CIRCEL_INSTALL_PACKAGE_DIR}")

# Keep this in sync with mlir/cmake/CMakeLists.txt!
set(MLIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir)
set(mlir_cmake_builddir "${MLIR_BINARY_DIR}/${MLIR_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
get_property(CIRCEL_EXPORTS GLOBAL PROPERTY CIRCEL_EXPORTS)
export(TARGETS ${CIRCEL_EXPORTS} FILE ${circel_cmake_builddir}/CIRCELTargets.cmake)

get_property(CIRCEL_ALL_LIBS GLOBAL PROPERTY CIRCEL_ALL_LIBS)
get_property(CIRCEL_DIALECT_LIBS GLOBAL PROPERTY CIRCEL_DIALECT_LIBS)
get_property(CIRCEL_CONVERSION_LIBS GLOBAL PROPERTY CIRCEL_CONVERSION_LIBS)
get_property(CIRCEL_TRANSLATION_LIBS GLOBAL PROPERTY CIRCEL_TRANSLATION_LIBS)
get_property(CIRCEL_ANALYSIS_LIBS GLOBAL PROPERTY CIRCEL_ANALYSIS_LIBS)
get_property(CIRCEL_VERIFICATION_LIBS GLOBAL PROPERTY CIRCEL_VERIFICATION_LIBS)

# Generate CIRCELConfig.cmake for the build tree.
set(CIRCEL_CONFIG_CMAKE_DIR "${circel_cmake_builddir}")
set(CIRCEL_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(CIRCEL_CONFIG_MLIR_CMAKE_DIR "${mlir_cmake_builddir}")
set(CIRCEL_CONFIG_LIBRARY_DIRS "${CIRCEL_LIBRARY_DIR}")
set(CIRCEL_CONFIG_BINARY_DIR "${CIRCEL_BINARY_DIR}")
set(CIRCEL_CONFIG_TOOLS_DIR "${CIRCEL_TOOLS_DIR}")
set(CIRCEL_CONFIG_INCLUDE_EXPORTS "include(\"\${CIRCEL_CMAKE_DIR}/CIRCELTargets.cmake\")")
set(CIRCEL_CONFIG_INCLUDE_DIRS
  "${CIRCEL_SOURCE_DIR}/include"
  "${CIRCEL_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/CIRCELConfig.cmake.in
  ${circel_cmake_builddir}/CIRCELConfig.cmake
  @ONLY)
set(CIRCEL_CONFIG_CMAKE_DIR)
set(CIRCEL_CONFIG_LLVM_CMAKE_DIR)
set(CIRCEL_CONFIG_MLIR_CMAKE_DIR)
set(CIRCEL_CONFIG_LIBRARY_DIRS)
set(CIRCEL_CONFIG_BINARY_DIR)
set(CIRCEL_CONFIG_TOOLS_DIR)
set(CIRCEL_CONFIG_INCLUDE_EXPORTS)
set(CIRCEL_CONFIG_INCLUDE_DIRS)

# Generate CIRCELConfig.cmake for the install tree.
set(CIRCEL_CONFIG_CODE "
# Compute the installation prefix from this CIRCELConfig.cmake file location.
get_filename_component(CIRCEL_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${CIRCEL_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(CIRCEL_CONFIG_CODE "${CIRCEL_CONFIG_CODE}
get_filename_component(CIRCEL_INSTALL_PREFIX \"\${CIRCEL_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(CIRCEL_CONFIG_CMAKE_DIR "\${CIRCEL_INSTALL_PREFIX}/${CIRCEL_INSTALL_PACKAGE_DIR}")
set(CIRCEL_CONFIG_LLVM_CMAKE_DIR "\${CIRCEL_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(CIRCEL_CONFIG_MLIR_CMAKE_DIR "\${CIRCEL_INSTALL_PREFIX}/${MLIR_INSTALL_PACKAGE_DIR}")
set(CIRCEL_CONFIG_LIBRARY_DIRS "\${CIRCEL_INSTALL_PREFIX}/lib")
set(CIRCEL_CONFIG_BINARY_DIR "\${CIRCEL_INSTALL_PREFIX}")
set(CIRCEL_CONFIG_TOOLS_DIR "\${CIRCEL_INSTALL_PREFIX}/bin")
set(CIRCEL_CONFIG_INCLUDE_EXPORTS "include(\"\${CIRCEL_CMAKE_DIR}/CIRCELTargets.cmake\")")
set(CIRCEL_CONFIG_INCLUDE_DIRS
  "\${CIRCEL_INSTALL_PREFIX}/include"
  )
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CIRCELConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CIRCELConfig.cmake
    @ONLY)
set(CIRCEL_CONFIG_CODE)
set(CIRCEL_CONFIG_CMAKE_DIR)
set(CIRCEL_CONFIG_LLVM_CMAKE_DIR)
set(CIRCEL_CONFIG_MLIR_CMAKE_DIR)
set(CIRCEL_CONFIG_LIBRARY_DIRS)
set(CIRCEL_CONFIG_BINARY_DIR)
set(CIRCEL_CONFIG_TOOLS_DIR)
set(CIRCEL_CONFIG_INCLUDE_EXPORTS)
set(CIRCEL_CONFIG_INCLUDE_DIRS)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the CIRCEL parts as well
  # Include the cmake files so other tools can use circel-tblgen, etc.
  install(EXPORT CIRCELTargets DESTINATION ${CIRCEL_INSTALL_PACKAGE_DIR}
          COMPONENT circel-cmake-exports)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/CIRCELConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddCIRCEL.cmake
    DESTINATION ${CIRCEL_INSTALL_PACKAGE_DIR}
    COMPONENT circel-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(circel-cmake-exports)
    add_llvm_install_targets(install-circel-cmake-exports
                             COMPONENT circel-cmake-exports)
  endif()
endif()
