add_mlir_jni_library(MLIRJNI
  JNI_SOURCE_PAIRS
    Attributes
    Block
    Builder
    Context
    Dialect
    Locations
    NativeObject
    Operations
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
    MLIRPass
    MLIRTransforms
    MLIRBindingsSupport
)
