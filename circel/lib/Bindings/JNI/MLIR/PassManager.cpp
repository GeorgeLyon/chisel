#include "circel/Bindings/JNI/MLIR/JNIBuilder.h"
#include "circel/Bindings/JNI/MLIR/JNIContext.h"
#include "circel/Bindings/JNI/MLIR/JNIPassManager.h"
#include "circel/Bindings/Support/ScopedPassManager.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "MLIR_PassManager.h"

using namespace circel;
using namespace mlir;

JNIPassManager::JNIPassManager(JNIEnv *env, jobject passManager)
    : JNIPassManager(env, JNIContext::unwrapPassManager(env, passManager)) {}

JNIEXPORT jobject JNICALL
Java_MLIR_PassManager_create(JNIEnv *env, jclass passManagerClass,
                             jclass passManagerSubclass, jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  auto passManager = ScopedPassManager(builder.getBuilder());

  /**
   PassManager does not use the JNI cache, since we want to instantiate a
   specific subclass as specified by the `passManagerClass` argument.
   */
  assert(env->IsAssignableFrom(passManagerSubclass, passManagerClass));
  auto methodID = env->GetMethodID(passManagerSubclass, "<init>", "(J)V");
  return env->NewObject(
      passManagerSubclass, methodID,
      reinterpret_cast<jlong>(passManager.toRetainedOpaquePointer().get()));
}

JNIEXPORT void JNICALL
Java_MLIR_PassManager_enableVerifier(JNIEnv *env, jobject jPassManager) {
  auto passManager = JNIPassManager(env, jPassManager);
  passManager.getScopedPassManager()->enableVerifier();
}

JNIEXPORT jboolean JNICALL Java_MLIR_PassManager_run(JNIEnv *env,
                                                     jobject jPassManager,
                                                     jobject jOp) {
  auto passManager = JNIPassManager(env, jPassManager);
  auto op = passManager.getBuilder().unwrap<Operation *>(jOp);
  auto result = passManager.getScopedPassManager().run(op);
  return result.succeeded();
}

JNIEXPORT void JNICALL
Java_MLIR_PassManager_addPrintIRPass(JNIEnv *env, jobject jPassManager) {
  auto passManager = JNIPassManager(env, jPassManager);
  passManager.getScopedPassManager()->addPass(mlir::createPrintIRPass());
}
