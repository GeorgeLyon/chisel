#include <circt/Dialect/FIRRTL/FIRRTLOps.h>

#include "circel/Bindings/JNI/MLIR/JNIBuilder.h"

#include "FIRRTL_Operations_Circuit.h"

using namespace circel;
using namespace circt::firrtl;

MLIR_JNI_DECLARE_CLASS_BINDING(CircuitOp)
MLIR_JNI_DEFINE_CLASS_BINDING(CircuitOp, "SIL/Operations$Constant")

JNIEXPORT jobject JNICALL Java_FIRRTL_Operations_00024Circuit_build(
    JNIEnv *env, jclass, jobject jBuilder, jobject jLoc, jstring jName) {
  auto builder = JNIBuilder(env, jBuilder);
  auto loc = builder.unwrap<mlir::Location>(jLoc);
  auto name = builder.unwrap<mlir::StringAttr>(jName);
  auto op = builder->create<CircuitOp>(loc, name);
  return builder.wrap(op);
}
