#include "MLIR_Values_Value.h"

#include "mlir-bindings/JNI/JNIBuilder.h"

#include "mlir/IR/Value.h"

using namespace mlir;
using namespace bindings;

// -- Value

MLIR_JNI_DEFINE_CLASS_BINDING(Value, "MLIR/Values$Value")

JNIEXPORT void JNICALL Java_MLIR_Value_dump(JNIEnv *env, jobject jvalue,
                                            jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  builder.unwrap<Value>(jvalue).dump();
}

// -- OpResult

MLIR_JNI_DEFINE_CLASS_BINDING(OpResult, "MLIR/Values$OpResult")
