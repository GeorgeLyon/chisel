#include "circel/Bindings/JNI/MLIR/JNIPassManager.h"

#include "FIRRTL_PassManager.h"
#include "circt/Firtool/Firtool.h"

#include <mlir/Pass/PassManager.h>

using namespace circel;
using namespace circt;

JNIEXPORT void JNICALL Java_FIRRTL_PassManager_addFirtoolPasses(
    JNIEnv *env, jobject jPassManager, jobject jOptions) {
  auto passManager = JNIPassManager(env, jPassManager);
  auto &pm = passManager.getPassManager();

  // Eventually, we can use JNI to access settings from "jOptions"
  static llvm::cl::OptionCategory mainCategory("firtool Options");
  firtool::FirtoolOptions options(mainCategory);

  if (failed(firtool::populatePreprocessTransforms(pm, options)))
    assert(false);

  if (failed(firtool::populateCHIRRTLToLowFIRRTL(pm, options, nullptr, "-")))
    assert(false);

  if (failed(firtool::populateLowFIRRTLToHW(pm, options)))
    assert(false);

  if (failed(firtool::populateHWToSV(pm, options)))
    assert(false);

  if (failed(firtool::populateExportVerilog(pm, options, llvm::outs())))
    assert(false);
}
