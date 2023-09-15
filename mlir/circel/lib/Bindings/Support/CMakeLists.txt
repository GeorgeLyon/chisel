add_mlir_library(MLIRBindingsSupport
    OpaquePointer.cpp
    ReferenceCountedPointer.cpp
    ScopedBuilder.cpp
    ScopedContext.cpp

    LINK_LIBS PRIVATE
    MLIRIR
    )
