add_subdirectory(Bindings)

# Must be added after `Bindings` as it hardcodes the classpath
add_subdirectory(circel-java-test-launcher)

# Get the classpath as an argument
get_property(MLIR_JAVA_CLASSPATH GLOBAL PROPERTY MLIR_JAVA_CLASSPATH)
list(JOIN MLIR_JAVA_CLASSPATH ":" MLIR_JAVA_CLASSPATH_ARGUMENT)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-circel "Running the CIRCEL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS 
    FileCheck count not
    circel-bindings-support-test
    circel-java-test-launcher
    circel-bindings-jni-mlir-test
    circel-bindings-swift-test
  )
set_target_properties(check-circel PROPERTIES FOLDER "Tests")
