#ifndef CIRCEL_BINDINGS_JNI_MLIR_JNI_CONTEXT_H_
#define CIRCEL_BINDINGS_JNI_MLIR_JNI_CONTEXT_H_

#include "circel/Bindings/Support/ScopedBuilder.h"
#include "circel/Bindings/Support/ScopedContext.h"
#include "circel/Bindings/Support/ScopedPassManager.h"

#include <jni.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/TypeRange.h>

namespace circel {

class JNIBuilder;

class JNIContext {
public:
  // -- Declaring which Java class corresponds to a C++ type

  struct JavaClass;
  static JavaClass &createJavaClass(const char *className);

  template <typename T> struct JavaClassBinding {
    static const JavaClass &getClass() = delete;
  };

  // -- Constructors

  JNIContext(JNIEnv *env, ScopedContext context) : env(env), context(context) {}
  JNIContext(JNIEnv *env, jobject context);

  // -- Wrapping

  template <typename T> jobject wrap(T value) const {
    return Wrapper<T>::wrap(*this, value);
  }

  // -- Unwrapping

  /// Unwraps a ScopedBuilder in situations where you don't have a JNIContext
  static ScopedBuilder unwrapBuilder(JNIEnv *env, jobject builder);

  /// Unwraps a PassManager in situations where you don't have a JNIContext
  static ScopedPassManager unwrapPassManager(JNIEnv *env, jobject passManager);

  template <typename CppValue, typename JavaValue>
  decltype(auto) unwrap(JavaValue javaValue) const {
    return Unwrapper<CppValue, JavaValue>::unwrap(*this, javaValue);
  }
  template <typename Element>
  llvm::SmallVector<Element> unwrapArray(jobjectArray objects) const {
    return unwrap<llvm::SmallVector<Element>, jobjectArray>(objects);
  }
  mlir::StringAttr unwrapString(jstring) const;

  // -- Accessors

  ScopedContext getContext() const { return context; }

  operator mlir::MLIRContext *() const { return context; }
  mlir::MLIRContext *operator->() const { return context; }

private:
  // -- Wrapper

  template <typename CppValue> struct Wrapper {
    static jobject wrap(const JNIContext &context, CppValue value) {
      return context.createJavaObjectFromOpaquePointer<CppValue>(
          context.getContext().wrap(value));
    };
  };
  template <> struct Wrapper<ScopedBuilder> {
    static jobject wrap(const JNIContext &context, ScopedBuilder builder) {
      return context.createJavaObjectFromOpaquePointer<ScopedBuilder>(
          builder.toRetainedOpaquePointer());
    };
  };

  // -- Unwrapper

  template <typename CppValue, typename JavaValue> struct Unwrapper {
    /**
     This is a fallback template, so if the C++ compiler is selecting it it
     probably means the bridge logic between the selected types has not been
     implemented.
     */
    static CppValue unwrap(const JNIContext &context,
                           JavaValue javaValue) = delete;
  };
  template <typename CppValue> struct Unwrapper<CppValue, jobject> {
    static CppValue unwrap(const JNIContext &context, jobject object) {
      return context.context.unwrap<CppValue>(
          context.getOpaquePointerFromJavaObject(object));
    }
  };
  template <typename CppValue>
  struct Unwrapper<llvm::SmallVector<CppValue>, jobjectArray> {
    static llvm::SmallVector<CppValue> unwrap(const JNIContext &context,
                                              jobjectArray objects) {
      jsize length = context.env->GetArrayLength(objects);
      llvm::SmallVector<CppValue> cppValues;
      cppValues.reserve(length);
      for (jsize i = 0, e = length; i < e; ++i) {
        auto object = context.env->GetObjectArrayElement(objects, i);
        cppValues.push_back(context.unwrap<CppValue>(object));
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
  template <> struct Unwrapper<mlir::StringAttr, jstring> {
    static mlir::StringAttr unwrap(const JNIContext &context, jstring string) {
      return context.unwrapString(string);
    }
  };

  // -- Conversion between Java Objects and OpaquePointers
  OpaquePointer getOpaquePointerFromJavaObject(jobject object) const;
  template <typename CppValue>
  jobject createJavaObjectFromOpaquePointer(OpaquePointer value) const {
    return createJavaObjectOfClassFromOpaquePointer(
        JavaClassBinding<CppValue>::getClass(), value);
  }
  jobject createJavaObjectOfClassFromOpaquePointer(const JavaClass &,
                                                   OpaquePointer) const;

  // -- Members

  friend class JNIBuilder;
  JNIEnv *env;
  ScopedContext context;
};

// -- Class Bindings

#define MLIR_JNI_DECLARE_CLASS_BINDING(CppType)                                \
  template <> struct ::circel::JNIContext::JavaClassBinding<CppType> {         \
    static const ::circel::JNIContext::JavaClass &getClass();                  \
  };
#define MLIR_JNI_DEFINE_CLASS_BINDING(CppType, JavaClassName)                  \
  const ::circel::JNIContext::JavaClass                                        \
      & ::circel::JNIContext::JavaClassBinding<CppType>::getClass() {          \
    static auto &clazz =                                                       \
        circel::JNIContext::createJavaClass("" JavaClassName "");              \
    return clazz;                                                              \
  }

// -- Declare as many bindings as possible here to avoid divergent behavior
// based on the set of header files included (Implementations reside in the
// corresponding cpp files).

MLIR_JNI_DECLARE_CLASS_BINDING(mlir::Attribute)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::IntegerAttr)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::StringAttr)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::ArrayAttr)

MLIR_JNI_DECLARE_CLASS_BINDING(mlir::Location)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::UnknownLoc)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::FileLineColLoc)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::CallSiteLoc)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::FusedLoc)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::NameLoc)

MLIR_JNI_DECLARE_CLASS_BINDING(mlir::Type)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::IntegerType)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::FunctionType)

MLIR_JNI_DECLARE_CLASS_BINDING(mlir::Block *)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::Region *)

MLIR_JNI_DECLARE_CLASS_BINDING(mlir::Value)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::BlockArgument)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::OpResult)

MLIR_JNI_DECLARE_CLASS_BINDING(mlir::OpState)
MLIR_JNI_DECLARE_CLASS_BINDING(mlir::ModuleOp)

MLIR_JNI_DECLARE_CLASS_BINDING(ScopedBuilder)

} // namespace circel

#endif // CIRCEL_BINDINGS_JNI_MLIR_JNI_CONTEXT_H_
