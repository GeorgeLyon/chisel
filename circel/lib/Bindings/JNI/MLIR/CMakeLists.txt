add_mlir_jni_library(CIRCELBindingsJNIMLIR
  JNI_SOURCE_PAIRS
    Attributes
    Block
    Builder
    Context
    Dialect
    Locations
    NativeObject
    Operations
    PassManager
    Region
    Types
    Values

  ADDITIONAL_JAVA_SOURCES
    NativeLibrary.java

  ADDITIONAL_CPP_SOURCES
    JNIDebugHelpers.cpp
    JNIVersion.cpp
  
  LINK_LIBS PRIVATE
    MLIRIR
    CIRCELBindingsSupport
)
