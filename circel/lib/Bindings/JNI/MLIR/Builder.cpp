#include "MLIR_Builder.h"

#include "circel/Bindings/JNI/MLIR/JNIBuilder.h"
#include "circel/Bindings/JNI/MLIR/JNIContext.h"
#include "circel/Bindings/Support/ScopedBuilder.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace circel;

MLIR_JNI_DEFINE_CLASS_BINDING(ScopedBuilder, "MLIR/Builder")

JNIBuilder::JNIBuilder(JNIEnv *env, jobject builder)
    : JNIBuilder(env, JNIContext::unwrapBuilder(env, builder)) {}

JNIEXPORT jobject JNICALL Java_MLIR_Builder_create(JNIEnv *env, jclass,
                                                   jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto scopedBuilder = ScopedBuilder(context.getContext());
  return context.wrap(scopedBuilder);
}

JNIEXPORT void JNICALL Java_MLIR_Builder_createBlock(JNIEnv *env,
                                                     jobject jBuilder,
                                                     jobject jRegion) {
  auto builder = JNIBuilder(env, jBuilder);
  auto region = builder.unwrap<mlir::Region *>(jRegion);
  builder->createBlock(region);
}

JNIEXPORT void JNICALL Java_MLIR_Builder_setInsertionPointToStart(
    JNIEnv *env, jobject jBuilder, jobject jBlock) {
  auto builder = JNIBuilder(env, jBuilder);
  auto *block = builder.unwrap<mlir::Block *>(jBlock);
  builder->setInsertionPointToStart(block);
}

JNIEXPORT void JNICALL Java_MLIR_Builder_setInsertionPointToEnd(
    JNIEnv *env, jobject jBuilder, jobject jBlock) {
  auto builder = JNIBuilder(env, jBuilder);
  auto *block = builder.unwrap<mlir::Block *>(jBlock);
  builder->setInsertionPointToEnd(block);
}

JNIEXPORT void JNICALL Java_MLIR_Builder_setInsertionPointBefore(
    JNIEnv *env, jobject jBuilder, jobject jOperation) {
  auto builder = JNIBuilder(env, jBuilder);
  auto *operation = builder.unwrap<mlir::Operation *>(jOperation);
  builder->setInsertionPoint(operation);
}

JNIEXPORT void JNICALL Java_MLIR_Builder_setInsertionPointAfter(
    JNIEnv *env, jobject jBuilder, jobject jOperation) {
  auto builder = JNIBuilder(env, jBuilder);
  auto *operation = builder.unwrap<mlir::Operation *>(jOperation);
  builder->setInsertionPointAfter(operation);
}
