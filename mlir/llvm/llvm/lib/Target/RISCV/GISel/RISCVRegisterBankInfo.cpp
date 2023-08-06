//===-- RISCVRegisterBankInfo.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the RegisterBankInfo class for RISC-V.
/// \todo This should be generated by TableGen.
//===----------------------------------------------------------------------===//

#include "RISCVRegisterBankInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterBank.h"
#include "llvm/CodeGen/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_TARGET_REGBANK_IMPL
#include "RISCVGenRegisterBank.inc"

using namespace llvm;

RISCVRegisterBankInfo::RISCVRegisterBankInfo(unsigned HwMode)
    : RISCVGenRegisterBankInfo(HwMode) {}
