#include "CircelJNI_IRBuilder.h"

#include "circel/Bindings/Support/Support.h"
#include "jni.h"

// -- IR Builder

static jfieldID nativeReferenceFieldID = nullptr;
static mlir::bindings::IRBuilder *unwrap(JNIEnv *env, jobject object) {
  return (mlir::bindings::IRBuilder *)env->GetLongField(object,
                                                        nativeReferenceFieldID);
}

// -- Methods

JNIEXPORT void JNICALL
Java_CircelJNI_IRBuilder_initialize(JNIEnv *env, jclass IRBuilderClass) {
  nativeReferenceFieldID =
      env->GetFieldID(IRBuilderClass, "nativeReference", "J");
}

JNIEXPORT jlong JNICALL Java_CircelJNI_IRBuilder_createNativeReference(JNIEnv *,
                                                                       jclass) {
  return (jlong) new mlir::bindings::IRBuilder();
}

JNIEXPORT void JNICALL Java_CircelJNI_IRBuilder_destroy(JNIEnv *env,
                                                        jobject object) {
  auto builder = unwrap(env, object);
  delete builder;
}

JNIEXPORT void JNICALL Java_CircelJNI_IRBuilder_test(JNIEnv *env,
                                                     jobject object) {
  auto builder = unwrap(env, object);
  builder->test();
}
