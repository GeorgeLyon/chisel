find_package(Java REQUIRED)
include(UseJava)

add_jar(mlir-bindings-jni-test
  SOURCES
  MLIRJNITests.java

  INCLUDE_JARS
  MLIRJNI_Jar)
get_target_property(JAR_FILE mlir-bindings-jni-test JAR_FILE)
set_property(GLOBAL APPEND PROPERTY MLIR_JAVA_CLASSPATH ${JAR_FILE})
add_dependencies(mlir-bindings-jni-test MLIRJNI)
