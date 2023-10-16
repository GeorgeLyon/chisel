#include "circel/Bindings/JNI/MLIR/JNIBuilder.h"

#include <mlir/IR/Value.h>

#include "MLIR_Values_Value.h"

using namespace circel;
using namespace mlir;

// -- Value

MLIR_JNI_DEFINE_CLASS_BINDING(Value, "MLIR/Values$Value")

JNIEXPORT void JNICALL Java_MLIR_Value_dump(JNIEnv *env, jobject jvalue,
                                            jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  builder.unwrap<Value>(jvalue).dump();
}

// -- OpResult

MLIR_JNI_DEFINE_CLASS_BINDING(OpResult, "MLIR/Values$OpResult")
