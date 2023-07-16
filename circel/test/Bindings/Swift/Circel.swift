// RUN check-swift-bindings | FileCheck %s

import CxxStdlib
import Circel

Circel.IRBuilder builder = Circel.IRBuilder()
// CHECK: "Hello, IRBuilder!"
builder.test()
