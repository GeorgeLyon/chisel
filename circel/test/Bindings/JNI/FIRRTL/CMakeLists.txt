find_package(Java REQUIRED)
include(UseJava)

add_jar(circel-bindings-jni-firrtl-test
  SOURCES
  FIRRTLJNITests.java

  INCLUDE_JARS
  MLIRJNI_Jar
  FIRRTLJNI_Jar)
get_target_property(JAR_FILE circel-bindings-jni-firrtl-test JAR_FILE)
set_property(GLOBAL APPEND PROPERTY MLIR_JAVA_CLASSPATH ${JAR_FILE})
add_dependencies(circel-bindings-jni-firrtl-test FIRRTLJNI)
