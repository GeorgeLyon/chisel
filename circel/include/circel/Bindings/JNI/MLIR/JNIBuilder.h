#ifndef CIRCEL_BINDINGS_JNI_MLIR_JNI_BUILDER_H_
#define CIRCEL_BINDINGS_JNI_MLIR_JNI_BUILDER_H_

#include "circel/Bindings/JNI/MLIR/JNIContext.h"
#include "circel/Bindings/Support/ScopedBuilder.h"

#include <jni.h>
#include <mlir/IR/Builders.h>

namespace circel {

class JNIBuilder {
public:
  // -- Constructors

  JNIBuilder(JNIEnv *env, ScopedBuilder builder) : env(env), builder(builder) {}
  JNIBuilder(JNIEnv *env, jobject builder);

  // -- Wrapping

  template <typename T> jobject wrap(T value) const {
    return getContext().createJavaObjectFromOpaquePointer<T>(
        builder.wrap(value));
  }

  // -- Unwrapping

  template <typename CppValue, typename JavaValue>
  decltype(auto) unwrap(JavaValue javaValue) const {
    return Unwrapper<CppValue, JavaValue>::unwrap(*this, javaValue);
  }
  template <typename Element>
  llvm::SmallVector<Element> unwrapArray(jobjectArray objects) const {
    return unwrap<llvm::SmallVector<Element>, jobjectArray>(objects);
  }

  // -- Accessors

  JNIContext getContext() const {
    return JNIContext(env, builder.getContext());
  }
  ScopedBuilder getBuilder() const { return builder; }

  ScopedBuilder operator->() const { return builder; }

private:
  // -- Unwrapper

  template <typename CppValue, typename JavaValue> struct Unwrapper {
    /**
     This is a fallback template, so if the C++ compiler is selecting it it
     probably means the bridge logic between the selected types has not been
     implemented.
     */
    static CppValue unwrap(const JNIBuilder &builder,
                           JavaValue javaValue) = delete;
  };
  template <typename CppValue> struct Unwrapper<CppValue, jobject> {
    static CppValue unwrap(const JNIBuilder &builder, jobject object) {
      return builder.builder.unwrap<CppValue>(
          builder.getContext().getOpaquePointerFromJavaObject(object));
    }
  };

  template <typename CppValue>
  struct Unwrapper<llvm::SmallVector<CppValue>, jobjectArray> {
    static llvm::SmallVector<CppValue> unwrap(const JNIBuilder &builder,
                                              jobjectArray objects) {
      jsize length = builder.env->GetArrayLength(objects);
      llvm::SmallVector<CppValue> cppValues;
      cppValues.reserve(length);
      for (jsize i = 0, e = length; i < e; ++i) {
        auto object = builder.env->GetObjectArrayElement(objects, i);
        cppValues.push_back(builder.unwrap<CppValue>(object));
      }
      return cppValues;
    }
  };

  template <typename CppType>
  struct Unwrapper<llvm::ArrayRef<CppType>, jobjectArray>
      : Unwrapper<llvm::SmallVector<CppType>, jobjectArray> {};

  template <>
  struct Unwrapper<mlir::TypeRange, jobjectArray>
      : Unwrapper<llvm::SmallVector<mlir::Type>, jobjectArray> {};

  template <>
  struct Unwrapper<mlir::ValueRange, jobjectArray>
      : Unwrapper<llvm::SmallVector<mlir::Value>, jobjectArray> {};

  template <> struct Unwrapper<mlir::StringAttr, jstring> {
    static mlir::StringAttr unwrap(const JNIBuilder &builder, jstring string) {
      return builder.getContext().unwrapString(string);
    }
  };

  // -- Members

  JNIEnv *env;
  ScopedBuilder builder;
};

} // namespace mlir

#endif // CIRCEL_BINDINGS_JNI_MLIR_JNI_BUILDER_H_
