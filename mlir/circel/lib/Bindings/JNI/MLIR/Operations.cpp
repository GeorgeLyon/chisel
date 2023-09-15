#include "MLIR_Operations_Module.h"
#include "MLIR_Operations_Operation.h"

#include "mlir-bindings/JNI/JNIBuilder.h"
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;
using namespace bindings;

// -- Operation

MLIR_JNI_DEFINE_CLASS_BINDING(mlir::OpState, "MLIR/Operations$Operation")

JNIEXPORT void JNICALL Java_MLIR_Operations_00024Operation_dump(
    JNIEnv *env, jobject op, jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  builder.unwrap<Operation *>(op)->dump();
}

JNIEXPORT jint JNICALL Java_MLIR_Operations_00024Operation_getNumberOfResults(
    JNIEnv *env, jobject jOperation, jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  return builder.unwrap<Operation *>(jOperation)->getNumResults();
}

JNIEXPORT jobject JNICALL Java_MLIR_Operations_00024Operation_getResult(
    JNIEnv *env, jobject jOperation, jobject jBuilder, jint index) {
  auto builder = JNIBuilder(env, jBuilder);
  auto result = builder.unwrap<Operation *>(jOperation)->getResult(index);
  return builder.wrap(result);
}

// -- Module

MLIR_JNI_DEFINE_CLASS_BINDING(mlir::ModuleOp, "MLIR/Operations$Module")

JNIEXPORT jobject JNICALL Java_MLIR_Operations_00024Module_build(
    JNIEnv *env, jclass, jobject jBuilder, jobject jLoc) {
  auto builder = JNIBuilder(env, jBuilder);
  auto loc = builder.unwrap<mlir::Location>(jLoc);
  auto op = builder->create<mlir::ModuleOp>(loc);
  return builder.wrap(op);
}

JNIEXPORT jobject JNICALL Java_MLIR_Operations_00024Module_getBody(
    JNIEnv *env, jobject jModule, jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  auto module = builder.unwrap<mlir::ModuleOp>(jModule);
  return builder.wrap(module.getBody());
}
