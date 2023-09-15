#include "MLIR_Types_FunctionType.h"
#include "MLIR_Types_IntegerType.h"
#include "MLIR_Types_Type.h"

#include "mlir-bindings/JNI/JNIContext.h"
#include <mlir/IR/BuiltinTypes.h>

using namespace mlir;
using namespace bindings;

// -- Type

MLIR_JNI_DEFINE_CLASS_BINDING(Type, "MLIR/Types$Type")

JNIEXPORT void JNICALL Java_MLIR_Types_00024Type_dump(JNIEnv *env,
                                                      jobject jType,
                                                      jobject jContext) {
  auto context = JNIContext(env, jContext);
  context.unwrap<Type>(jType).dump();
}

// -- IntegerType

MLIR_JNI_DEFINE_CLASS_BINDING(IntegerType, "MLIR/Types$IntegerType")

JNIEXPORT jobject JNICALL Java_MLIR_Types_00024IntegerType_get(JNIEnv *env,
                                                               jclass,
                                                               jobject jContext,
                                                               jint width) {
  auto context = JNIContext(env, jContext);
  return context.wrap(IntegerType::get(context, width));
}

// -- FunctionType

MLIR_JNI_DEFINE_CLASS_BINDING(FunctionType, "MLIR/Types$FunctionType")

JNIEXPORT jobject JNICALL Java_MLIR_Types_00024FunctionType_get(
    JNIEnv *env, jclass, jobject jContext, jobjectArray jInputs,
    jobjectArray jResults) {
  auto context = JNIContext(env, jContext);
  auto inputs = context.unwrapArray<Type>(jInputs);
  auto results = context.unwrapArray<Type>(jResults);
  return context.wrap(FunctionType::get(context, inputs, results));
}
