#ifndef MLIR_BINDINGS_JNI_JNI_DEBUG_HELPERS_H_
#define MLIR_BINDINGS_JNI_JNI_DEBUG_HELPERS_H_

#include <jni.h>
#include <ostream>

namespace mlir {
namespace bindings {
class DebugJavaClassName {
  JNIEnv *env;
  jclass clazz;

public:
  void streamTo(std::ostream &os) const;
};
std::ostream &operator<<(std::ostream &os, DebugJavaClassName const &name);

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_JNI_JNI_DEBUG_HELPERS_H_
