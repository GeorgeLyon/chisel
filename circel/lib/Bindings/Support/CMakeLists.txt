add_circel_library(CIRCELBindingsSupport
    OpaquePointer.cpp
    ReferenceCountedPointer.cpp
    ScopedBuilder.cpp
    ScopedContext.cpp
    ScopedPassManager.cpp

    LINK_LIBS PRIVATE
    MLIRIR
    MLIRPass
    MLIRTransforms
    )
