#include <circt/Dialect/FIRRTL/FIRRTLOps.h>

#include "circel/Bindings/JNI/MLIR/JNIBuilder.h"

#include "FIRRTL_Operations_Circuit.h"
#include "FIRRTL_Operations_Module.h"
#include "FIRRTL_Operations_Register.h"

using namespace circel;
using namespace circt::firrtl;

// -- Circuit

MLIR_JNI_DECLARE_CLASS_BINDING(CircuitOp)
MLIR_JNI_DEFINE_CLASS_BINDING(CircuitOp, "FIRRTL/Operations$Circuit")

JNIEXPORT jobject JNICALL Java_FIRRTL_Operations_00024Circuit_build(
    JNIEnv *env, jclass, jobject jBuilder, jobject jLoc, jstring jName) {
  auto builder = JNIBuilder(env, jBuilder);
  auto loc = builder.unwrap<mlir::Location>(jLoc);
  auto name = builder.unwrap<mlir::StringAttr>(jName);
  auto op = builder->create<CircuitOp>(loc, name);
  return builder.wrap(op);
}

JNIEXPORT jobject JNICALL Java_FIRRTL_Operations_00024Circuit_getBody(
    JNIEnv *env, jobject jOp, jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  auto op = builder.unwrap<CircuitOp>(jOp);
  auto region = &op.getRegion();
  auto block = &region->front();
  return builder.wrap(block);
}

// -- Module

MLIR_JNI_DECLARE_CLASS_BINDING(FModuleOp)
MLIR_JNI_DEFINE_CLASS_BINDING(FModuleOp, "FIRRTL/Operations$Module")

JNIEXPORT jobject JNICALL Java_FIRRTL_Operations_00024Module_build(
    JNIEnv *env, jclass, jobject jBuilder, jobject jLoc, jstring jName,
    jobject jConvention) {
  auto builder = JNIBuilder(env, jBuilder);
  auto loc = builder.unwrap<mlir::Location>(jLoc);
  auto name = builder.unwrap<mlir::StringAttr>(jName);
  auto convention = builder.unwrap<circt::firrtl::ConventionAttr>(jConvention);
  // Ports are added via `addPort`
  auto ports = llvm::SmallVector<PortInfo>{};
  auto op = builder->create<FModuleOp>(loc, name, convention, ports);
  return builder.wrap(op);
}

JNIEXPORT jobject JNICALL Java_FIRRTL_Operations_00024Module_addPort(
    JNIEnv *env, jobject jOp, jobject jBuilder, jstring jName, jobject jType,
    jboolean directionIsIn) {
  auto builder = JNIBuilder(env, jBuilder);
  auto op = builder.unwrap<FModuleOp>(jOp);
  auto name = builder.unwrap<mlir::StringAttr>(jName);
  auto type = builder.unwrap<mlir::Type>(jType);
  auto direction = directionIsIn ? Direction::In : Direction::Out;
  auto port = PortInfo{name, type, direction};
  auto portIndex = op.getNumPorts();
  auto portsToInsert = llvm::SmallVector<std::pair<unsigned int, PortInfo>, 1>{
      std::make_pair(portIndex, port)};
  op.insertPorts(portsToInsert);
  auto arg = op.getArgument(portIndex);
  return builder.wrap(arg);
}

JNIEXPORT jobject JNICALL Java_FIRRTL_Operations_00024Module_getBody(
    JNIEnv *env, jobject jOp, jobject jBuilder) {
  auto builder = JNIBuilder(env, jBuilder);
  auto op = builder.unwrap<FModuleOp>(jOp);
  auto region = &op.getRegion();
  auto block = &region->front();
  return builder.wrap(block);
}

// -- Register

MLIR_JNI_DECLARE_CLASS_BINDING(RegOp)
MLIR_JNI_DEFINE_CLASS_BINDING(RegOp, "FIRRTL/Operations$Register")

/*
public static native Register build(Builder builder, Location location, Type
elementType, Value clock, String name, NameKind nameKind, boolean isForceable);
        */
JNIEXPORT jobject JNICALL Java_FIRRTL_Operations_00024Register_build(
    JNIEnv *env, jclass, jobject jBuilder, jobject jLoc, jobject jType,
    jobject jValue, jstring jName, jobject jNameKind, jboolean isForceable) {
  auto builder = JNIBuilder(env, jBuilder);
  auto loc = builder.unwrap<mlir::Location>(jLoc);
  auto type = builder.unwrap<mlir::Type>(jType);
  auto value = builder.unwrap<mlir::Value>(jValue);
  auto name = builder.unwrap<mlir::StringAttr>(jName);
  auto nameKind = builder.unwrap<circt::firrtl::NameKindEnumAttr>(jNameKind);
  auto annotations =
      builder->getArrayAttr(llvm::SmallVector<mlir::Attribute>{});
  auto innerSymAttr = circt::hw::InnerSymAttr();
  auto forceableAttr = isForceable ? mlir::UnitAttr() : nullptr;
  auto op = builder->create<RegOp>(loc, type, value, name, nameKind,
                                   annotations, innerSymAttr, forceableAttr);
  return builder.wrap(op);
}
