#include "FIRRTL_Dialect.h"

#include "circel/Bindings/JNI/MLIR/JNIContext.h"

#include <circt/Dialect/FIRRTL/FIRRTLDialect.h>
#include <jni.h>

#include "FIRRTL_Dialect.h"

using namespace circel;

JNIEXPORT void JNICALL Java_FIRRTL_Dialect_load(JNIEnv *env, jclass,
                                                jobject jContext) {
  auto context = JNIContext(env, jContext);
  context->getOrLoadDialect<circt::firrtl::FIRRTLDialect>();
}
