add_mlir_jni_library(FIRRTLJNI
  JNI_SOURCE_PAIRS
    Attributes
    Dialect
    Operations
    Types

  ADDITIONAL_JAVA_SOURCES
    NativeLibrary.java

  ADDITIONAL_CPP_SOURCES
  
  JNI_DEPENDENCIES
    MLIRJNI

  LINK_LIBS 
    PUBLIC
      CIRCTFIRRTL
    PRIVATE
      MLIRIR
      CIRCELBindingsSupport
)
