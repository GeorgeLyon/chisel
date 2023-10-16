#include "circel/Bindings/JNI/MLIR/JNIContext.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

#include "MLIR_Attributes_ArrayAttribute.h"
#include "MLIR_Attributes_Attribute.h"
#include "MLIR_Attributes_IntegerAttribute.h"
#include "MLIR_Attributes_StringAttribute.h"

using namespace circel;

// -- Attribute

MLIR_JNI_DEFINE_CLASS_BINDING(mlir::Attribute, "MLIR/Attributes$Attribute")

JNIEXPORT void JNICALL Java_MLIR_Attributes_00024Attribute_dump(
    JNIEnv *env, jobject jAttribute, jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto attribute = context.unwrap<mlir::Attribute>(jAttribute);
  attribute.dump();
}

// -- IntegerAttr

MLIR_JNI_DEFINE_CLASS_BINDING(mlir::IntegerAttr,
                              "MLIR/Attributes$IntegerAttribute")

JNIEXPORT jobject JNICALL Java_MLIR_Attributes_00024IntegerAttribute_get(
    JNIEnv *env, jclass, jobject jContext, jobject jType, jlong value) {
  auto context = JNIContext(env, jContext);
  auto type = context.unwrap<mlir::Type>(jType);
  auto attr = mlir::Builder(context).getIntegerAttr(type, value);
  auto object = context.wrap(attr);
  return object;
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

// -- ArrayAttr

MLIR_JNI_DEFINE_CLASS_BINDING(mlir::ArrayAttr, "MLIR/Attributes$ArrayAttribute")

JNIEXPORT jobject JNICALL Java_MLIR_Attributes_00024ArrayAttribute_get(
    JNIEnv *env, jclass, jobject jContext, jobjectArray jMembers) {
  auto context = JNIContext(env, jContext);
  auto members = context.unwrapArray<mlir::Attribute>(jMembers);
  auto arrayAttr = mlir::ArrayAttr::get(context, members);
  return context.wrap(arrayAttr);
}
