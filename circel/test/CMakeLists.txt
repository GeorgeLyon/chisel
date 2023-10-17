add_subdirectory(Bindings)

# Must be added after `Bindings` as it hardcodes the classpath
add_subdirectory(circel-java-test-launcher)

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
    circel-bindings-jni-firrtl-test
    circel-bindings-swift-test
  )
set_target_properties(check-circel PROPERTIES FOLDER "Tests")
