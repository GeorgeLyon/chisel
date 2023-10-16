#include <cassert>
#include <iostream>
#include <jni.h>
#include <string>

#include <llvm/Support/CommandLine.h>

using namespace llvm;

cl::opt<std::string> className("class", cl::desc("Class to run"), cl::Required);

JNIEXPORT jint JNICALL JNI_OnLoad_MLIRJNI(JavaVM *vm, void *reserved) {
  return JNI_VERSION_10;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  std::string classpath = MLIR_JAVA_CLASSPATH;
  JNIEnv *env = nullptr;

  // -- Initialize the JVM
  {
    JavaVM *jvm = nullptr;
    JavaVMInitArgs vm_args;
    vm_args.version = JNI_VERSION_10;
    vm_args.ignoreUnrecognized = JNI_FALSE;
    JavaVMOption options[1];
    vm_args.nOptions = 0;
    vm_args.options = options;
    auto result = JNI_CreateJavaVM(&jvm, (void **)&env, &vm_args);
    assert(result == JNI_OK && "Failed to create JVM");
  }

  // We don't create a local frame here because the process terminates at the
  // end of this method, meaning leaking references is OK.

  // -- Covert classpath strings to URL objects
  jobjectArray classpathURLs;
  {
    // -- Parse classpath into array of strings
    jclass stringClass;
    jobjectArray classpathStrings;
    {
      auto classpathString = env->NewStringUTF(classpath.c_str());
      auto splitString = env->NewStringUTF(":");

      stringClass = env->FindClass("java/lang/String");
      assert(stringClass != nullptr && "Failed to find java/lang/String");
      jmethodID splitMethod = env->GetMethodID(
          stringClass, "split", "(Ljava/lang/String;)[Ljava/lang/String;");
      assert(splitMethod != nullptr && "Failed to find String.split");
      classpathStrings = (jobjectArray)env->CallObjectMethod(
          classpathString, splitMethod, splitString);
      assert(classpathStrings != nullptr);
    }

    // Get URL Class and constructor
    jclass URLClass = env->FindClass("java/net/URL");
    assert(URLClass != nullptr && "Failed to find java/net/URL");
    jmethodID URLConstructor =
        env->GetMethodID(URLClass, "<init>", "(Ljava/lang/String;)V");
    assert(URLConstructor != nullptr && "Failed to find URL constructor");

    // Get String.format method and format string
    jmethodID formatMethod =
        env->GetStaticMethodID(stringClass, "format",
                               "(Ljava/lang/String;[Ljava/lang/Object;)"
                               "Ljava/lang/String;");
    jstring formatString = env->NewStringUTF("file://%s");

    classpathURLs = env->NewObjectArray(env->GetArrayLength(classpathStrings),
                                        URLClass, nullptr);
    assert(classpathURLs != nullptr && "Failed to initialize URL array");
    for (int i = 0; i < env->GetArrayLength(classpathStrings); ++i) {
      auto classpathString =
          (jstring)env->GetObjectArrayElement(classpathStrings, i);

      // Format URL
      auto formatArguments =
          env->NewObjectArray(1, stringClass, classpathString);
      auto classpathURLString =
          reinterpret_cast<jstring>(env->CallStaticObjectMethod(
              stringClass, formatMethod, formatString, formatArguments));

      auto classpathURL =
          env->NewObject(URLClass, URLConstructor, classpathURLString);
      env->SetObjectArrayElement(classpathURLs, i, classpathURL);
    }

    // Create classloader
    jclass urlClassLoaderClass = env->FindClass("java/net/URLClassLoader");
    assert(urlClassLoaderClass && "Failed to find URLClassLoader class");
    jmethodID urlClassLoaderConstructor =
        env->GetMethodID(urlClassLoaderClass, "<init>", "([Ljava/net/URL;)V");
    auto urlClassLoaderObject = env->NewObject(
        urlClassLoaderClass, urlClassLoaderConstructor, classpathURLs);

    // Find requested class
    auto urlClassLoaderFindClassMethodID =
        env->GetMethodID(urlClassLoaderClass, "findClass",
                         "(Ljava/lang/String;)Ljava/lang/Class;");
    jstring jclassName = env->NewStringUTF(className.c_str());
    jclass methodClass = reinterpret_cast<jclass>(env->CallObjectMethod(
        urlClassLoaderObject, urlClassLoaderFindClassMethodID, jclassName));
    assert(methodClass && "Failed to find method class");

    // Call `main` method
    jmethodID mainMethodID =
        env->GetStaticMethodID(methodClass, "main", "([Ljava/lang/String;)V");
    assert(mainMethodID && "Failed to find main method");
    auto mainMethodArgs = env->NewObjectArray(0, stringClass, nullptr);
    env->CallStaticVoidMethod(methodClass, mainMethodID, mainMethodArgs);

    env->ExceptionDescribe();
  }

  return 0;
}
