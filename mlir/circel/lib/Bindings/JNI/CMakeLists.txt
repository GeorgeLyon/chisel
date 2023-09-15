function(add_mlir_jni_library name)
  cmake_parse_arguments(ARG
    ""
    ""
    "JNI_SOURCE_PAIRS;ADDITIONAL_JAVA_SOURCES;ADDITIONAL_CPP_SOURCES;JNI_DEPENDENCIES;LINK_LIBS"
    ${ARGN})

  find_package(Java REQUIRED)
  find_package(JNI REQUIRED)
  include(UseJava)

  set(JNI_JAVA_SOURCES ${ARG_JNI_SOURCE_PAIRS})
  list(TRANSFORM JNI_JAVA_SOURCES APPEND .java)

  set(JNI_CPP_SOURCES ${ARG_JNI_SOURCE_PAIRS})
  list(TRANSFORM JNI_CPP_SOURCES APPEND .cpp)

  set(JAR_DEPENDENCIES ${ARG_JNI_DEPENDENCIES})
  list(TRANSFORM JAR_DEPENDENCIES APPEND _Jar)

  add_jar(${name}_Jar
    ${JNI_JAVA_SOURCES}
    ${ARG_ADDITIONAL_JAVA_SOURCES}

    INCLUDE_JARS
    ${JAR_DEPENDENCIES}

    GENERATE_NATIVE_HEADERS ${name}_Headers)
  add_mlir_library(${name}
    SHARED

    ${JNI_CPP_SOURCES}
    ${ARG_ADDITIONAL_CPP_SOURCES}

    EXCLUDE_FROM_LIBMLIR
    DISABLE_INSTALL
    
    LINK_LIBS 
    ${ARG_LINK_LIBS}
    ${ARG_JNI_DEPENDENCIES}
    ${name}_Headers
    )
  set_target_properties(${name} PROPERTIES
    BUILD_WITH_INSTALL_RPATH OFF
    BUILD_RPATH "."
    INSTALL_RPATH "."
  )
  set_target_properties(${name} PROPERTIES SUFFIX ".jni")
  target_include_directories(${name} PRIVATE ${JNI_INCLUDE_DIRS})

  # Add to the global classpath
  get_target_property(JAR_FILE ${name}_Jar JAR_FILE)
  set_property(GLOBAL APPEND PROPERTY MLIR_JAVA_CLASSPATH ${JAR_FILE})
endfunction()

add_subdirectory(MLIR)
