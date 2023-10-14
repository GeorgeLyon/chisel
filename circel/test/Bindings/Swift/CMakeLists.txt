set(CMAKE_Swift_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/swift)
enable_language(Swift)

add_executable(circel-bindings-swift-test
  Test.swift
)
target_link_libraries(circel-bindings-swift-test CIRCELBindingsSupport)
target_compile_options(circel-bindings-swift-test
  PRIVATE
    -emit-module
    "SHELL:-cxx-interoperability-mode=swift-5.9"
    "SHELL:-Xfrontend -validate-tbd-against-ir=none"
    -Xcc -std=c++17)
