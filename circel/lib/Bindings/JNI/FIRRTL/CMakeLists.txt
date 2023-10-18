add_mlir_jni_library(FIRRTLJNI
  JNI_SOURCE_PAIRS
    Attributes
    Dialect
    Operations
    PassManager
    Types

  ADDITIONAL_JAVA_SOURCES
    NativeLibrary.java
    FirtoolOptions.java

  ADDITIONAL_CPP_SOURCES
  
  JNI_DEPENDENCIES
    MLIRJNI

  LINK_LIBS 
    PUBLIC
      CIRCTFIRRTL
      CIRCTFirtool
    PRIVATE
      MLIRIR
      CIRCELBindingsSupport
)
