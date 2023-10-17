find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)


add_llvm_tool(circel-java-test-launcher
  circel-java-test-launcher.cpp)
target_link_libraries(circel-java-test-launcher PRIVATE 
  MLIRJNI
  LLVMSupport
  JNI::JVM
  JNI::JNI)
target_include_directories(circel-java-test-launcher PRIVATE ${JNI_INCLUDE_DIRS})

get_property(MLIR_JAVA_CLASSPATH GLOBAL PROPERTY MLIR_JAVA_CLASSPATH)
list(JOIN MLIR_JAVA_CLASSPATH ":" MLIR_JAVA_CLASSPATH_ARGUMENT)
target_compile_definitions(circel-java-test-launcher PRIVATE MLIR_JAVA_CLASSPATH="${MLIR_JAVA_CLASSPATH_ARGUMENT}")

llvm_update_compile_flags(circel-java-test-launcher)

mlir_check_all_link_libraries(circel-java-test-launcher)
