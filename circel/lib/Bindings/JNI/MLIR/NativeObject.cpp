#include "MLIR_NativeObject.h"
#include "circel/Bindings/Support/OpaquePointer.h"

#include <cassert>
#include <jni.h>

using namespace circel;

JNIEXPORT void JNICALL
Java_MLIR_NativeObject_releaseNativeObject(JNIEnv *, jclass, jlong pointer) {
  assert(pointer != 0);
  assert(sizeof(pointer) == sizeof(void *));
  OpaquePointer(reinterpret_cast<void *>(pointer)).releaseUnderlyingResource();
}
