#include "MLIR_Context.h"
#include "mlir-bindings/JNI/JNIContext.h"
#include "mlir-bindings/Support/ScopedBuilder.h"
#include "mlir-bindings/Support/ScopedContext.h"

using namespace mlir::bindings;

// -- Java Class Binding

struct JNIContext::JavaClass {
public:
  using ID = uint16_t;
  ID id;
  const char *name;

  JavaClass(const char *name)
      : id([] {
          std::unique_lock<std::shared_mutex> lock(idLock);
          auto id = nextID++;
          assert(id != std::numeric_limits<ID>::max());
          return id;
        }()),
        name(name) {}
  static ID getMaxID() {
    std::shared_lock<std::shared_mutex> lock(idLock);
    return nextID;
  }

private:
  static std::shared_mutex idLock;
  static JavaClass::ID nextID;
};
std::shared_mutex JNIContext::JavaClass::idLock;
JNIContext::JavaClass::ID JNIContext::JavaClass::nextID = 0;

JNIContext::JavaClass &JNIContext::createJavaClass(const char *name) {
  return *(new JavaClass(name));
}

// -- JNI Cache

/**
 The context owns a cache of JNI values that are used for wrapping and
 unwrapping Java objects.
 */
class JNICache {
public:
  // -- Java Object Constructor

  struct Constructor {
    jclass clazz;
    jmethodID initializer;

    Constructor() : clazz(nullptr), initializer(nullptr) {}
    Constructor(JNIEnv *env, const char *className) {
      clazz = globalReferenceToClass(env, className);
      initializer = env->GetMethodID(clazz, "<init>", "(J)V");
    }

    jobject createJavaObject(JNIEnv *env, OpaquePointer pointer) {
      return env->NewObject(clazz, initializer,
                            reinterpret_cast<jlong>(pointer.get()));
    }
  };

  // -- Common JNI values

  Constructor contextConstructor;
  Constructor builderConstructor;
  jclass nativeObjectClass;
  jfieldID nativeObjectPointerField;

  // -- Converting between OpaquePointer and Java Objects

  OpaquePointer getOpaquePointerFromJavaObject(JNIEnv *env, jobject object) {
    return OpaquePointer(reinterpret_cast<void *>(
        env->GetLongField(object, nativeObjectPointerField)));
  }

  jobject getJavaObjectFromOpaquePointer(JNIEnv *env,
                                         const JNIContext::JavaClass &javaClass,
                                         OpaquePointer pointer) {
    {
      std::shared_lock<std::shared_mutex> readLock(constructorsMutex);

      if (javaClass.id < constructors.size() &&
          constructors[javaClass.id].clazz != nullptr) {
        return constructors[javaClass.id].createJavaObject(env, pointer);
      }
    }
    // If we reached this point, we did not find an entry for this class and the
    // cache is currently unlocked (so it may be initialized concurrently).
    std::unique_lock<std::shared_mutex> writeLock(constructorsMutex);
    if (javaClass.id >= constructors.size()) {
      constructors.resize(javaClass.id + 1);
    }
    if (constructors[javaClass.id].clazz == nullptr) {
      constructors[javaClass.id] = Constructor(env, javaClass.name);
    }
    return constructors[javaClass.id].createJavaObject(env, pointer);
  }

  // -- Validation

  bool isValidForContextClass(JNIEnv *env, jclass jContextClass) {
    return env->IsSameObject(jContextClass, contextConstructor.clazz);
  }
  bool isValidForBuilderClass(JNIEnv *env, jclass jBuilderClass) {
    return env->IsSameObject(jBuilderClass, builderConstructor.clazz);
  }

  // -- Lifecycle

  JNICache(JNIEnv *env)
      : contextConstructor(env, "MLIR/Context"),
        builderConstructor(env, "MLIR/Builder"),
        nativeObjectClass(globalReferenceToClass(env, "MLIR/NativeObject")),
        nativeObjectPointerField(
            env->GetFieldID(nativeObjectClass, "pointer", "J")),
        vm([&] {
          JavaVM *vm;
          env->GetJavaVM(&vm);
          return vm;
        }()),
        constructors(JNIContext::JavaClass::getMaxID()) {}

  ~JNICache() {
    JNIEnv *env;
    vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_10);
    env->DeleteGlobalRef(contextConstructor.clazz);
    env->DeleteGlobalRef(builderConstructor.clazz);
    env->DeleteGlobalRef(nativeObjectClass);
    for (auto &constructor : constructors) {
      if (constructor.clazz) {
        env->DeleteGlobalRef(constructor.clazz);
      }
    }
  }

  // JNICache cannot be copied or moved.
  JNICache(const JNICache &) = delete;
  JNICache &operator=(const JNICache &) = delete;
  JNICache(JNICache &&) = delete;
  JNICache &operator=(JNICache &&) = delete;

private:
  // -- Convenience

  static jclass globalReferenceToClass(JNIEnv *env, const char *name) {
    auto localClass = env->FindClass(name);
    auto globalClass = static_cast<jclass>(env->NewGlobalRef(localClass));
    env->DeleteLocalRef(localClass);
    return globalClass;
  }

  // -- Private Members

  JavaVM *vm;
  std::shared_mutex constructorsMutex;
  std::vector<Constructor> constructors;
};

// -- Context-owned JNI Cache

struct ContextOwnedJNICache {
  std::shared_ptr<JNICache> cache;
};

static std::shared_ptr<JNICache> JNIContextGetCache(JNIContext context) {
  return static_cast<ContextOwnedJNICache *>(context.getContext().getUserData())
      ->cache;
}

static void JNIContextSetCache(ScopedContext context,
                               std::shared_ptr<JNICache> cache) {
  auto *container = new ContextOwnedJNICache{cache};
  return context.setUserData(container, [](void *data) {
    delete static_cast<ContextOwnedJNICache *>(data);
  });
}

// -- Shared JNI Cache

/**
 We use a static shared reference to the JNI cache for operations which don't
 have a Context (unwrapping a context or builder).
 Updating this pointer is protected by a mutex
 */

class SharedJNICache {
  std::shared_mutex mutex;
  std::weak_ptr<JNICache> reference;

public:
  /// @returns A shared pointer to the shared cache. May be null.
  std::shared_ptr<JNICache> get() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return reference.lock();
  }
  void set(std::shared_ptr<JNICache> cache) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    reference = cache;
  }
};
static SharedJNICache sharedCache;

// -- Construction

JNIContext::JNIContext(JNIEnv *env, jobject jContext)
    : env(env), context([&] {
        auto jContextClass = env->GetObjectClass(jContext);
        auto cache = sharedCache.get();
        // If this assertion trips, consult the comment in
        // `Java_MLIR_Context_create` below.
        assert(cache);
        assert(cache->isValidForContextClass(env, jContextClass));
        auto opaquePointer =
            cache->getOpaquePointerFromJavaObject(env, jContext);
        return ScopedContext::getFromOpaquePointer(opaquePointer);
      }()) {}

// -- Unwrapping

ScopedBuilder JNIContext::unwrapBuilder(JNIEnv *env, jobject builder) {
  auto jBuilderClass = env->GetObjectClass(builder);
  auto cache = sharedCache.get();
  // If this assertion trips, consult the comment in
  // `Java_MLIR_Context_create` below.
  assert(cache);
  assert(cache->isValidForBuilderClass(env, jBuilderClass));
  auto opaquePointer = cache->getOpaquePointerFromJavaObject(env, builder);
  return ScopedBuilder::getFromOpaquePointer(opaquePointer);
}

mlir::StringAttr JNIContext::unwrapString(jstring string) const {
  auto cstr = env->GetStringUTFChars(string, nullptr);
  auto length = env->GetStringUTFLength(string);
  auto attr = mlir::StringAttr::get(context, mlir::StringRef(cstr, length));
  env->ReleaseStringUTFChars(string, cstr);
  return attr;
}

// -- Unwrapping

OpaquePointer JNIContext::getOpaquePointerFromJavaObject(jobject object) const {
  auto cache = JNIContextGetCache(*this);
  return cache->getOpaquePointerFromJavaObject(env, object);
}

// -- Wrapping

jobject JNIContext::createJavaObjectOfClassFromOpaquePointer(
    const struct JavaClass &javaClass, OpaquePointer pointer) const {
  auto cache = JNIContextGetCache(*this);
  return cache->getJavaObjectFromOpaquePointer(env, javaClass, pointer);
}

// -- JNI

JNIEXPORT jobject JNICALL Java_MLIR_Context_create(JNIEnv *env,
                                                   jclass jContextClass) {
  auto scopedContext = ScopedContext::create();

  auto cache = ::sharedCache.get();
  if (!cache) {
    // We are initializing the cache for the first time
    cache = std::make_shared<JNICache>(env);
    sharedCache.set(cache);
  } else {
    /**
    Currently, the only situation we know of where the shared cache becomes
    invalid is when running multiple times in a single JVM instance (as
    happens when run multiple times from a single SBT invocation). Because the
    shared cache pointer is weak, the old cache should be cleaned up but this is
    not guaranteed. Currently, we `close()` all of our Contexts when we are done
    with them, which should clean up the cache. If this assertion trips, that
    likely means a Context object has survived from the previous run, which may
    be an error. If this is not an error (i.e. we decide we want a static
    Context which is never explicitly closed), we can change this code to simply
    create a new cache... though we should still validate that we aren't
    thrashing the cache if for some reason the old classes are still calling
    into the library.
    We could evolve the caching strategy in a number of ways depending on what
    use cases we actually want to support. For example, we could:
     - Just replace the cache every time it is invalid
     - Give each cache an id and only replace the shared cache with higher-id
    caches
    */
    assert(cache->isValidForContextClass(env, jContextClass));
  }

  /// Set the context cache
  JNIContextSetCache(scopedContext, cache);

  return cache->contextConstructor.createJavaObject(
      env, scopedContext.toRetainedOpaquePointer());
}
