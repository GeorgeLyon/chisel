#ifndef CIRCEL_BINDINGS_JNI_JNI_DEBUG_HELPERS_H_
#define CIRCEL_BINDINGS_JNI_JNI_DEBUG_HELPERS_H_

#include <jni.h>
#include <ostream>

namespace circel {
class DebugJavaClassName {
  JNIEnv *env;
  jclass clazz;

public:
  void streamTo(std::ostream &os) const;
};
std::ostream &operator<<(std::ostream &os, DebugJavaClassName const &name);

} // namespace circel

#endif // CIRCEL_BINDINGS_JNI_JNI_DEBUG_HELPERS_H_
