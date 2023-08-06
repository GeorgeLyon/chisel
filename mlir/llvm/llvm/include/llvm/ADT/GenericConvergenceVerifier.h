//===- GenericConvergenceVerifier.h ---------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// A verifier for the static rules of convergence control tokens that works
/// with both LLVM IR and MIR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_GENERICCONVERGENCEVERIFIER_H
#define LLVM_ADT_GENERICCONVERGENCEVERIFIER_H

#include "llvm/ADT/GenericCycleInfo.h"

namespace llvm {

template <typename ContextT> class GenericConvergenceVerifier {
public:
  using BlockT = typename ContextT::BlockT;
  using FunctionT = typename ContextT::FunctionT;
  using ValueRefT = typename ContextT::ValueRefT;
  using InstructionT = typename ContextT::InstructionT;
  using DominatorTreeT = typename ContextT::DominatorTreeT;
  using CycleInfoT = GenericCycleInfo<ContextT>;
  using CycleT = typename CycleInfoT::CycleT;

  void initialize(raw_ostream *OS,
                  function_ref<void(const Twine &Message)> FailureCB,
                  const FunctionT &F) {
    clear();
    this->OS = OS;
    this->FailureCB = FailureCB;
    Context = ContextT(&F);
  }

  void clear();
  void visit(const InstructionT &I);
  void verify(const DominatorTreeT &DT);

  bool sawTokens() const { return ConvergenceKind == ControlledConvergence; }

private:
  raw_ostream *OS;
  std::function<void(const Twine &Message)> FailureCB;
  DominatorTreeT *DT;
  CycleInfoT CI;
  ContextT Context;

  /// Whether the current function has convergencectrl operand bundles.
  enum {
    ControlledConvergence,
    UncontrolledConvergence,
    NoConvergence
  } ConvergenceKind = NoConvergence;

  // Cache token uses found so far. Note that we track the unique definitions
  // and not the token values.
  DenseMap<const InstructionT *, const InstructionT *> Tokens;

  const InstructionT *findAndCheckConvergenceTokenUsed(const InstructionT &I);
  bool isControlledConvergent(const InstructionT &I);
  bool isConvergent(const InstructionT &I) const;

  void reportFailure(const Twine &Message, ArrayRef<Printable> Values);
};

} // end namespace llvm

#endif // LLVM_ADT_GENERICCONVERGENCEVERIFIER_H
