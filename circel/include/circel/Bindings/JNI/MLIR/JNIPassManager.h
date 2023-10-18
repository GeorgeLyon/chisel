#ifndef CIRCEL_BINDINGS_JNI_MLIR_JNI_PASS_MANAGER_H_
#define CIRCEL_BINDINGS_JNI_MLIR_JNI_PASS_MANAGER_H_

#include "circel/Bindings/JNI/MLIR/JNIBuilder.h"
#include "circel/Bindings/Support/ScopedPassManager.h"

#include <jni.h>
#include <mlir/IR/Builders.h>

namespace circel {

class JNIPassManager {
public:
  JNIPassManager(JNIEnv *env, ScopedPassManager passManager)
      : env(env), passManager(std::move(passManager)) {}
  JNIPassManager(JNIEnv *env, jobject passManager);

  // -- Accessors

  JNIBuilder getBuilder() const {
    return JNIBuilder(env, passManager.getBuilder());
  }

  mlir::PassManager &getPassManager() { return passManager.getPassManager(); }
  ScopedPassManager getScopedPassManager() const { return passManager; }

private:
  // -- Members

  JNIEnv *env;
  ScopedPassManager passManager;
};

} // namespace circel

#endif // CIRCEL_BINDINGS_JNI_MLIR_JNI_PASS_MANAGER_H_
