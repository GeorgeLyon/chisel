#include "circel/Bindings/JNI/MLIR/JNIContext.h"

#include <circt/Dialect/FIRRTL/FIRRTLTypes.h>

#include "FIRRTL_Types_Clock.h"

using namespace circel;
using namespace circt::firrtl;

// -- Clock

MLIR_JNI_DECLARE_CLASS_BINDING(ClockType)
MLIR_JNI_DEFINE_CLASS_BINDING(ClockType, "FIRRTL/Types$Clock")

JNIEXPORT jobject JNICALL Java_FIRRTL_Types_00024Clock_get(JNIEnv *env, jclass,
                                                           jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto type = ClockType::get(context);
  return context.wrap(type);
}
