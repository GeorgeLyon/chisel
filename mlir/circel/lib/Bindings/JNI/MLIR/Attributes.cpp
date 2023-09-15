#include "mlir-bindings/JNI/JNIContext.h"
#include <mlir/IR/BuiltinAttributes.h>

#include "MLIR_Attributes_Attribute.h"
#include "MLIR_Attributes_StringAttribute.h"

using namespace mlir::bindings;

// -- Attribute

MLIR_JNI_DEFINE_CLASS_BINDING(mlir::Attribute, "MLIR/Attributes$Attribute")

JNIEXPORT void JNICALL Java_MLIR_Attributes_00024Attribute_dump(
    JNIEnv *env, jobject jAttribute, jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto attribute = context.unwrap<mlir::Attribute>(jAttribute);
  attribute.dump();
}

// -- StringAttr

MLIR_JNI_DEFINE_CLASS_BINDING(mlir::StringAttr,
                              "MLIR/Attributes$StringAttribute")

JNIEXPORT jobject JNICALL Java_MLIR_Attributes_00024StringAttribute_get(
    JNIEnv *env, jclass, jobject jContext, jstring value) {
  auto context = JNIContext(env, jContext);
  auto stringAttr = context.unwrap<mlir::StringAttr>(value);
  return context.wrap(stringAttr);
}
