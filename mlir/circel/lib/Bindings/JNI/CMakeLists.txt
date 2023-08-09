find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)

add_jar(CircelJNI_Java
    NativeObject.java
    GENERATE_NATIVE_HEADERS CircelJNIHeaders)
add_circt_library(CircelJNI
    SHARED

    NativeObject.cpp
    
    LINK_LIBS PRIVATE
    CIRCTFirtool
    MLIRBindingsSupport
    )
target_link_libraries(CircelJNI PRIVATE CircelJNIHeaders)
target_include_directories(CircelJNI PRIVATE ${JNI_INCLUDE_DIRS})
