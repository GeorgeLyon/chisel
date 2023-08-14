//===- FunctionPropertiesAnalysis.cpp - Function Properties Analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionPropertiesInfo and FunctionPropertiesAnalysis
// classes used to extract function properties.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include <deque>

using namespace llvm;

cl::opt<bool> EnableDetailedFunctionProperties(
    "enable-detailed-function-properties", cl::Hidden, cl::init(false),
    cl::desc("Whether or not to compute detailed function properties."));

cl::opt<unsigned> BigBasicBlockInstructionThreshold(
    "big-basic-block-instruction-threshold", cl::Hidden, cl::init(500),
    cl::desc("The minimum number of instructions a basic block should contain "
             "before being considered big."));

cl::opt<unsigned> MediumBasicBlockInstructionThreshold(
    "medium-basic-block-instruction-threshold", cl::Hidden, cl::init(15),
    cl::desc("The minimum number of instructions a basic block should contain "
             "before being considered medium-sized."));

namespace {
int64_t getNrBlocksFromCond(const BasicBlock &BB) {
  int64_t Ret = 0;
  if (const auto *BI = dyn_cast<BranchInst>(BB.getTerminator())) {
    if (BI->isConditional())
      Ret += BI->getNumSuccessors();
  } else if (const auto *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
    Ret += (SI->getNumCases() + (nullptr != SI->getDefaultDest()));
  }
  return Ret;
}

int64_t getUses(const Function &F) {
  return ((!F.hasLocalLinkage()) ? 1 : 0) + F.getNumUses();
}
} // namespace

void FunctionPropertiesInfo::reIncludeBB(const BasicBlock &BB) {
  updateForBB(BB, +1);
}

void FunctionPropertiesInfo::updateForBB(const BasicBlock &BB,
                                         int64_t Direction) {
  assert(Direction == 1 || Direction == -1);
  BasicBlockCount += Direction;
  BlocksReachedFromConditionalInstruction +=
      (Direction * getNrBlocksFromCond(BB));
  for (const auto &I : BB) {
    if (auto *CS = dyn_cast<CallBase>(&I)) {
      const auto *Callee = CS->getCalledFunction();
      if (Callee && !Callee->isIntrinsic() && !Callee->isDeclaration())
        DirectCallsToDefinedFunctions += Direction;
    }
    if (I.getOpcode() == Instruction::Load) {
      LoadInstCount += Direction;
    } else if (I.getOpcode() == Instruction::Store) {
      StoreInstCount += Direction;
    }
  }
  TotalInstructionCount += Direction * BB.sizeWithoutDebug();

  if (EnableDetailedFunctionProperties) {
    unsigned SuccessorCount = succ_size(&BB);
    if (SuccessorCount == 1)
      BasicBlocksWithSingleSuccessor += Direction;
    else if (SuccessorCount == 2)
      BasicBlocksWithTwoSuccessors += Direction;
    else if (SuccessorCount > 2)
      BasicBlocksWithMoreThanTwoSuccessors += Direction;

    unsigned PredecessorCount = pred_size(&BB);
    if (PredecessorCount == 1)
      BasicBlocksWithSinglePredecessor += Direction;
    else if (PredecessorCount == 2)
      BasicBlocksWithTwoPredecessors += Direction;
    else if (PredecessorCount > 2)
      BasicBlocksWithMoreThanTwoPredecessors += Direction;

    if (TotalInstructionCount > BigBasicBlockInstructionThreshold)
      BigBasicBlocks += Direction;
    else if (TotalInstructionCount > MediumBasicBlockInstructionThreshold)
      MediumBasicBlocks += Direction;
    else
      SmallBasicBlocks += Direction;

    for (const Instruction &I : BB.instructionsWithoutDebug()) {
      if (I.isCast())
        CastInstructionCount += Direction;

      if (I.getType()->isFloatTy())
        FloatingPointInstructionCount += Direction;
      else if (I.getType()->isIntegerTy())
        IntegerInstructionCount += Direction;

      for (unsigned int OperandIndex = 0; OperandIndex < I.getNumOperands();
           ++OperandIndex) {
        if (const Constant *C =
                dyn_cast<Constant>(I.getOperand(OperandIndex))) {
          if (C->getType()->isIntegerTy())
            IntegerConstantCount += Direction;
          else if (C->getType()->isFloatTy())
            FloatingPointConstantCount += Direction;
        }
      }
    }
  }
}

void FunctionPropertiesInfo::updateAggregateStats(const Function &F,
                                                  const LoopInfo &LI) {

  Uses = getUses(F);
  TopLevelLoopCount = llvm::size(LI);
  MaxLoopDepth = 0;
  std::deque<const Loop *> Worklist;
  llvm::append_range(Worklist, LI);
  while (!Worklist.empty()) {
    const auto *L = Worklist.front();
    MaxLoopDepth =
        std::max(MaxLoopDepth, static_cast<int64_t>(L->getLoopDepth()));
    Worklist.pop_front();
    llvm::append_range(Worklist, L->getSubLoops());
  }
}

FunctionPropertiesInfo FunctionPropertiesInfo::getFunctionPropertiesInfo(
    Function &F, FunctionAnalysisManager &FAM) {
  return getFunctionPropertiesInfo(F, FAM.getResult<DominatorTreeAnalysis>(F),
                                   FAM.getResult<LoopAnalysis>(F));
}

FunctionPropertiesInfo FunctionPropertiesInfo::getFunctionPropertiesInfo(
    const Function &F, const DominatorTree &DT, const LoopInfo &LI) {

  FunctionPropertiesInfo FPI;
  for (const auto &BB : F)
    if (DT.isReachableFromEntry(&BB))
      FPI.reIncludeBB(BB);
  FPI.updateAggregateStats(F, LI);
  return FPI;
}

void FunctionPropertiesInfo::print(raw_ostream &OS) const {
  OS << "BasicBlockCount: " << BasicBlockCount << "\n"
     << "BlocksReachedFromConditionalInstruction: "
     << BlocksReachedFromConditionalInstruction << "\n"
     << "Uses: " << Uses << "\n"
     << "DirectCallsToDefinedFunctions: " << DirectCallsToDefinedFunctions
     << "\n"
     << "LoadInstCount: " << LoadInstCount << "\n"
     << "StoreInstCount: " << StoreInstCount << "\n"
     << "MaxLoopDepth: " << MaxLoopDepth << "\n"
     << "TopLevelLoopCount: " << TopLevelLoopCount << "\n"
     << "TotalInstructionCount: " << TotalInstructionCount << "\n";
  if (EnableDetailedFunctionProperties) {
    OS << "BasicBlocksWithSingleSuccessor: " << BasicBlocksWithSingleSuccessor
       << "\n"
       << "BasicBlocksWithTwoSuccessors: " << BasicBlocksWithTwoSuccessors
       << "\n"
       << "BasicBlocksWithMoreThanTwoSuccessors: "
       << BasicBlocksWithMoreThanTwoSuccessors << "\n"
       << "BasicBlocksWithSinglePredecessor: "
       << BasicBlocksWithSinglePredecessor << "\n"
       << "BasicBlocksWithTwoPredecessors: " << BasicBlocksWithTwoPredecessors
       << "\n"
       << "BasicBlocksWithMoreThanTwoPredecessors: "
       << BasicBlocksWithMoreThanTwoPredecessors << "\n"
       << "BigBasicBlocks: " << BigBasicBlocks << "\n"
       << "MediumBasicBlocks: " << MediumBasicBlocks << "\n"
       << "SmallBasicBlocks: " << SmallBasicBlocks << "\n"
       << "CastInstructionCount: " << CastInstructionCount << "\n"
       << "FloatingPointInstructionCount: " << FloatingPointInstructionCount
       << "\n"
       << "IntegerInstructionCount: " << IntegerInstructionCount << "\n"
       << "IntegerConstantCount: " << IntegerConstantCount << "\n"
       << "FloatingPointConstantCount: " << FloatingPointConstantCount << "\n";
  }
  OS << "\n";
}

AnalysisKey FunctionPropertiesAnalysis::Key;

FunctionPropertiesInfo
FunctionPropertiesAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  return FunctionPropertiesInfo::getFunctionPropertiesInfo(F, FAM);
}

PreservedAnalyses
FunctionPropertiesPrinterPass::run(Function &F, FunctionAnalysisManager &AM) {
  OS << "Printing analysis results of CFA for function "
     << "'" << F.getName() << "':"
     << "\n";
  AM.getResult<FunctionPropertiesAnalysis>(F).print(OS);
  return PreservedAnalyses::all();
}

FunctionPropertiesUpdater::FunctionPropertiesUpdater(
    FunctionPropertiesInfo &FPI, CallBase &CB)
    : FPI(FPI), CallSiteBB(*CB.getParent()), Caller(*CallSiteBB.getParent()) {
  assert(isa<CallInst>(CB) || isa<InvokeInst>(CB));
  // For BBs that are likely to change, we subtract from feature totals their
  // contribution. Some features, like max loop counts or depths, are left
  // invalid, as they will be updated post-inlining.
  SmallPtrSet<const BasicBlock *, 4> LikelyToChangeBBs;
  // The CB BB will change - it'll either be split or the callee's body (single
  // BB) will be pasted in.
  LikelyToChangeBBs.insert(&CallSiteBB);

  // The caller's entry BB may change due to new alloca instructions.
  LikelyToChangeBBs.insert(&*Caller.begin());

  // The successors may become unreachable in the case of `invoke` inlining.
  // We track successors separately, too, because they form a boundary, together
  // with the CB BB ('Entry') between which the inlined callee will be pasted.
  Successors.insert(succ_begin(&CallSiteBB), succ_end(&CallSiteBB));

  // Inlining only handles invoke and calls. If this is an invoke, and inlining
  // it pulls another invoke, the original landing pad may get split, so as to
  // share its content with other potential users. So the edge up to which we
  // need to invalidate and then re-account BB data is the successors of the
  // current landing pad. We can leave the current lp, too - if it doesn't get
  // split, then it will be the place traversal stops. Either way, the
  // discounted BBs will be checked if reachable and re-added.
  if (const auto *II = dyn_cast<InvokeInst>(&CB)) {
    const auto *UnwindDest = II->getUnwindDest();
    Successors.insert(succ_begin(UnwindDest), succ_end(UnwindDest));
  }

  // Exclude the CallSiteBB, if it happens to be its own successor (1-BB loop).
  // We are only interested in BBs the graph moves past the callsite BB to
  // define the frontier past which we don't want to re-process BBs. Including
  // the callsite BB in this case would prematurely stop the traversal in
  // finish().
  Successors.erase(&CallSiteBB);

  for (const auto *BB : Successors)
    LikelyToChangeBBs.insert(BB);

  // Commit the change. While some of the BBs accounted for above may play dual
  // role - e.g. caller's entry BB may be the same as the callsite BB - set
  // insertion semantics make sure we account them once. This needs to be
  // followed in `finish`, too.
  for (const auto *BB : LikelyToChangeBBs)
    FPI.updateForBB(*BB, -1);
}

void FunctionPropertiesUpdater::finish(FunctionAnalysisManager &FAM) const {
  // Update feature values from the BBs that were copied from the callee, or
  // might have been modified because of inlining. The latter have been
  // subtracted in the FunctionPropertiesUpdater ctor.
  // There could be successors that were reached before but now are only
  // reachable from elsewhere in the CFG.
  // One example is the following diamond CFG (lines are arrows pointing down):
  //    A
  //  /   \
  // B     C
  // |     |
  // |     D
  // |     |
  // |     E
  //  \   /
  //    F
  // There's a call site in C that is inlined. Upon doing that, it turns out
  // it expands to
  //   call void @llvm.trap()
  //   unreachable
  // F isn't reachable from C anymore, but we did discount it when we set up
  // FunctionPropertiesUpdater, so we need to re-include it here.
  // At the same time, D and E were reachable before, but now are not anymore,
  // so we need to leave D out (we discounted it at setup), and explicitly
  // remove E.
  SetVector<const BasicBlock *> Reinclude;
  SetVector<const BasicBlock *> Unreachable;
  const auto &DT =
      FAM.getResult<DominatorTreeAnalysis>(const_cast<Function &>(Caller));

  if (&CallSiteBB != &*Caller.begin())
    Reinclude.insert(&*Caller.begin());

  // Distribute the successors to the 2 buckets.
  for (const auto *Succ : Successors)
    if (DT.isReachableFromEntry(Succ))
      Reinclude.insert(Succ);
    else
      Unreachable.insert(Succ);

  // For reinclusion, we want to stop at the reachable successors, who are at
  // the beginning of the worklist; but, starting from the callsite bb and
  // ending at those successors, we also want to perform a traversal.
  // IncludeSuccessorsMark is the index after which we include successors.
  const auto IncludeSuccessorsMark = Reinclude.size();
  bool CSInsertion = Reinclude.insert(&CallSiteBB);
  (void)CSInsertion;
  assert(CSInsertion);
  for (size_t I = 0; I < Reinclude.size(); ++I) {
    const auto *BB = Reinclude[I];
    FPI.reIncludeBB(*BB);
    if (I >= IncludeSuccessorsMark)
      Reinclude.insert(succ_begin(BB), succ_end(BB));
  }

  // For exclusion, we don't need to exclude the set of BBs that were successors
  // before and are now unreachable, because we already did that at setup. For
  // the rest, as long as a successor is unreachable, we want to explicitly
  // exclude it.
  const auto AlreadyExcludedMark = Unreachable.size();
  for (size_t I = 0; I < Unreachable.size(); ++I) {
    const auto *U = Unreachable[I];
    if (I >= AlreadyExcludedMark)
      FPI.updateForBB(*U, -1);
    for (const auto *Succ : successors(U))
      if (!DT.isReachableFromEntry(Succ))
        Unreachable.insert(Succ);
  }

  const auto &LI = FAM.getResult<LoopAnalysis>(const_cast<Function &>(Caller));
  FPI.updateAggregateStats(Caller, LI);
}

bool FunctionPropertiesUpdater::isUpdateValid(Function &F,
                                              const FunctionPropertiesInfo &FPI,
                                              FunctionAnalysisManager &FAM) {
  DominatorTree DT(F);
  LoopInfo LI(DT);
  auto Fresh = FunctionPropertiesInfo::getFunctionPropertiesInfo(F, DT, LI);
  return FPI == Fresh;
}
