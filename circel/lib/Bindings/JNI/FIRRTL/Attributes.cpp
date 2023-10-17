#include "circel/Bindings/JNI/MLIR/JNIContext.h"

#include <circt/Dialect/FIRRTL/FIRRTLTypes.h>

#include "FIRRTL_Attributes_Convention.h"

using namespace circel;
using namespace circt::firrtl;

// -- Convention

MLIR_JNI_DECLARE_CLASS_BINDING(ConventionAttr)
MLIR_JNI_DEFINE_CLASS_BINDING(ConventionAttr, "FIRRTL/Attributes$Convention")

JNIEXPORT jobject JNICALL Java_FIRRTL_Attributes_00024Convention_getScalarized(
    JNIEnv *env, jclass, jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto attr = ConventionAttr::get(context, Convention::Scalarized);
  return context.wrap(attr);
}
