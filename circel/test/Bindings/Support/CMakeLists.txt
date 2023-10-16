add_circel_executable(circel-bindings-support-test
  Test.cpp)
target_link_libraries(circel-bindings-support-test
  PRIVATE
  CIRCELBindingsSupport)
