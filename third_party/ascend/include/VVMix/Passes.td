/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_VVMIX_PASSES
#define TRITON_ADAPTER_VVMIX_PASSES

include "mlir/Pass/PassBase.td"

def VVMix : Pass<"vv-mix", "mlir::ModuleOp"> {
    let summary = "VVMix pass for SIMT scope tensor_ptr handling and outlining";
    let constructor = "triton::createVVMixPass()";
    let options = [
        Option<"outline_simt_scope", "outline-simt-scope", "bool", "false",
           "Experimental feature">
    ];
    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::func::FuncDialect",
        "mlir::triton::TritonDialect",
        "mlir::scf::SCFDialect",
        "mlir::tensor::TensorDialect",
        "mlir::triton::ascend::TritonAscendDialect"
    ];
}

#endif // TRITON_ADAPTER_VVMIX_PASSES