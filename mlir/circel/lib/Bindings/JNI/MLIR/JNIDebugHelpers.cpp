
#include "JNIDebugHelpers.h"
#include "jni.h"
#include <cassert>

using namespace mlir::bindings;

void DebugJavaClassName::streamTo(std::ostream &os) const {
  jclass classClass = env->FindClass("java/lang/Class");
  jmethodID getNameID =
      env->GetMethodID(classClass, "getName", "()Ljava/lang/String;");
  jstring string =
      reinterpret_cast<jstring>(env->CallObjectMethod(clazz, getNameID));
  const char *cstr = env->GetStringUTFChars(string, nullptr);
  os << std::string(cstr);
  env->ReleaseStringUTFChars(string, cstr);
}

std::ostream &operator<<(std::ostream &os, DebugJavaClassName const &name) {
  name.streamTo(os);
  return os;
}
