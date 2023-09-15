#include "MLIR_Locations_CallSite.h"
#include "MLIR_Locations_FileLineColumn.h"
#include "MLIR_Locations_Fused.h"
#include "MLIR_Locations_Location.h"
#include "MLIR_Locations_Name.h"
#include "MLIR_Locations_Unknown.h"

#include "mlir-bindings/JNI/JNIContext.h"

using namespace mlir;
using namespace bindings;

MLIR_JNI_DEFINE_CLASS_BINDING(Location, "MLIR/Locations$Location")

JNIEXPORT void JNICALL Java_MLIR_Locations_00024Location_dump(
    JNIEnv *env, jobject jLocation, jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto location = context.unwrap<mlir::Location>(jLocation);
  location.dump();
}

MLIR_JNI_DEFINE_CLASS_BINDING(CallSiteLoc, "MLIR/Locations$CallSite")

JNIEXPORT jobject JNICALL Java_MLIR_Locations_00024CallSite_get(
    JNIEnv *env, jclass, jobject jContext, jobject jCallee, jobject jCaller) {
  auto context = JNIContext(env, jContext);
  auto callee = context.unwrap<mlir::Location>(jCallee);
  auto caller = context.unwrap<mlir::Location>(jCaller);
  auto loc = mlir::CallSiteLoc::get(callee, caller);
  return context.wrap(loc);
}

MLIR_JNI_DEFINE_CLASS_BINDING(FileLineColLoc, "MLIR/Locations$FileLineColumn")

JNIEXPORT jobject JNICALL Java_MLIR_Locations_00024FileLineColumn_get(
    JNIEnv *env, jclass, jobject jContext, jstring jFile, jint jLine,
    jint jColumn) {
  auto context = JNIContext(env, jContext);
  auto file = context.unwrap<StringAttr>(jFile);
  auto loc = FileLineColLoc::get(context, file, jLine, jColumn);
  return context.wrap(loc);
}

MLIR_JNI_DEFINE_CLASS_BINDING(FusedLoc, "MLIR/Locations$Fused")

JNIEXPORT jobject JNICALL Java_MLIR_Locations_00024Fused_get(
    JNIEnv *env, jclass, jobject jContext, jobjectArray jLocs) {
  auto context = JNIContext(env, jContext);
  auto locs = context.unwrapArray<mlir::Location>(jLocs);
  auto loc = FusedLoc::get(context, locs);
  return context.wrap(loc);
}

MLIR_JNI_DEFINE_CLASS_BINDING(NameLoc, "MLIR/Locations$Name")

JNIEXPORT jobject JNICALL Java_MLIR_Locations_00024Name_get(JNIEnv *env, jclass,
                                                            jobject jContext,
                                                            jstring jName,
                                                            jobject jChild) {
  auto context = JNIContext(env, jContext);
  auto name = context.unwrap<StringAttr>(jName);
  auto child = context.unwrap<mlir::Location>(jChild);
  auto loc = NameLoc::get(name, child);
  return context.wrap(loc);
}

MLIR_JNI_DEFINE_CLASS_BINDING(UnknownLoc, "MLIR/Locations$Unknown")

JNIEXPORT jobject JNICALL
Java_MLIR_Locations_00024Unknown_get(JNIEnv *env, jclass, jobject jContext) {
  auto context = JNIContext(env, jContext);
  auto loc = UnknownLoc::get(context);
  return context.wrap(loc);
}
