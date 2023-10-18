#include "circel/Bindings/JNI/MLIR/JNIContext.h"

#include <circt/Dialect/FIRRTL/FIRRTLTypes.h>

#include "FIRRTL_Attributes_Convention.h"
#include "FIRRTL_Attributes_NameKind.h"

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

// -- NameKind

MLIR_JNI_DECLARE_CLASS_BINDING(NameKindEnumAttr)
MLIR_JNI_DEFINE_CLASS_BINDING(NameKindEnumAttr, "FIRRTL/Attributes$NameKind")

JNIEXPORT jobject JNICALL Java_FIRRTL_Attributes_00024NameKind_getDroppable(
    JNIEnv *env, jclass, jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto attr = NameKindEnumAttr::get(context, NameKindEnum::DroppableName);
  return context.wrap(attr);
}

JNIEXPORT jobject JNICALL Java_FIRRTL_Attributes_00024NameKind_getInteresting(
    JNIEnv *env, jclass, jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto attr = NameKindEnumAttr::get(context, NameKindEnum::InterestingName);
  return context.wrap(attr);
}
