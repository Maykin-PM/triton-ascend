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

#include "ascend/include/VVMix/VVMixPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vv-mix"

using namespace mlir;
using namespace triton;

// Rewrite tensor_ptr to ptr_tensor in simt scope;
static LogicalResult processTensorPtrInSimtScope(ModuleOp moduleOp) {

  bool outlineTensorPtr = false;
  SmallVector<scope::ScopeOp> scopeOps;
  moduleOp.walk([&](scope::ScopeOp op) {
    if (auto vectorType = op->getAttrOfType<StringAttr>("vector_type")) {
      if (vectorType.str() != "simt") {
        return;
      }
    }
    scopeOps.push_back(op);
  });

  for (auto scopeOp : scopeOps) {
    SmallVector<Operation *> lsOpsX;
    SmallVector<Operation *> lsOps;
    scopeOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<triton::LoadOp, triton::StoreOp>(op)) {
        bool crossScope = false;
        for (auto &opr : op->getOpOperands()) {
          auto val = opr.get();
          if (auto *defOp = val.getDefiningOp()) {
            if (!isa<triton::MakeTensorPtrOp>(defOp))
              continue;
            if (!scopeOp->isAncestor(defOp))
              crossScope = true;
            break;
          }
        }
        if (crossScope)
          lsOpsX.insert(lsOpsX.end(), op);

        lsOps.insert(lsOps.end(), op);
      }
    });

    DenseMap<Value, Value> cachedPtrOffset;
    for (auto op : lsOps) {
      auto tensor_ptr = op->getOperand(0);
      if (auto ptrType = dyn_cast<triton::PointerType>(tensor_ptr.getType())) {
        if (auto tensorType = dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {

          auto defOp = tensor_ptr.getDefiningOp<triton::MakeTensorPtrOp>();
          assert(defOp);
          OpBuilder builder(defOp);
          Location loc = defOp.getLoc();

          auto base = defOp.getBase();
          auto shape = defOp.getShape();
          auto strides = defOp.getStrides();
          auto offsets = defOp.getOffsets();
          auto tensorShape = tensorType.getShape();

          auto indexTensorType = RankedTensorType::get(tensorShape, builder.getI64Type());

          Value ptr_offset; 
          if (cachedPtrOffset.count(tensor_ptr))
            ptr_offset = cachedPtrOffset[tensor_ptr];
          else {
            for (unsigned i = 0; i < tensorShape.size(); ++i) {
              auto indexI32RowType =
                  RankedTensorType::get({tensorShape[i]}, builder.getI32Type());
              auto indexRowType =
                  RankedTensorType::get({tensorShape[i]}, builder.getI64Type());

              auto i64Offsets = builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), offsets[i]);
              Value splatOffset =
                  builder.create<triton::SplatOp>(loc, indexRowType, i64Offsets);
              Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                                tensorShape[i]);
              Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

              Value offsetWithRange =
                  builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
              for (uint j = 0; j < tensorShape.size(); ++j) {
                if (j == i)
                  continue;
                offsetWithRange =
                    builder.create<triton::ExpandDimsOp>(loc, offsetWithRange, j);
              }
              Value splatStride = builder.create<triton::SplatOp>(
                  loc, offsetWithRange.getType(), strides[i]);
              Value offsetWithStride =
                  builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
              Value broadcasted = builder.create<triton::BroadcastOp>(
                  loc, indexTensorType, offsetWithStride);
              if (i == 0) {
                ptr_offset = broadcasted;
              } else {
                ptr_offset = builder.create<arith::AddIOp>(loc, ptr_offset, broadcasted);
              }
            }
            cachedPtrOffset[tensor_ptr] = ptr_offset;
          }

          std::optional<ArrayRef<int>> boundaryCheck;
          if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
            assert(!loadOp.getMask() && !loadOp.getOther());
            boundaryCheck = loadOp.getBoundaryCheck();
          } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
            assert(!storeOp.getMask());
            boundaryCheck = storeOp.getBoundaryCheck();
          }

          IRRewriter rewriter(op->getContext());
          rewriter.setInsertionPoint(op);
          loc = op->getLoc();

          // get newPtr
          Value newPtr;
          auto basePtrType = cast<triton::PointerType>(base.getType());
          auto ptrTensorType = RankedTensorType::get(tensorShape, basePtrType);
          if (outlineTensorPtr) {
            newPtr = rewriter.create<triton::SplatOp>(loc, ptrTensorType, base);
            newPtr = rewriter.create<triton::AddPtrOp>(loc, ptrTensorType, newPtr, ptr_offset);
          }

          // get mask
          Value newMask;
          bool inited = false;
          if (boundaryCheck.has_value()) {
            auto maskTensorType =
              RankedTensorType::get(tensorShape, builder.getI1Type());
            for (auto i : boundaryCheck.value()) {
              // Add range
              auto indexI32RowType =
                  RankedTensorType::get({tensorShape[i]}, builder.getI32Type());
              auto indexRowType =
                  RankedTensorType::get({tensorShape[i]}, builder.getI64Type());
              auto i64Offsets = builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), offsets[i]);
              Value splatOffset =
                  builder.create<triton::SplatOp>(loc, indexRowType, i64Offsets);
              Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                                tensorShape[i]);
              Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

              // Expand dimensions
              Value offsetWithRange =
                  builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
              for (uint j = 0; j < tensorShape.size(); ++j) {
                if (j == i)
                  continue;
                offsetWithRange =
                    builder.create<triton::ExpandDimsOp>(loc, offsetWithRange, j);
              }

              // Compare with lower bound
              Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
                  loc, builder.getI64Type(), 0);
              Value splatLowerBound = builder.create<triton::SplatOp>(
                  loc, offsetWithRange.getType(), lowerBound);
              Value cmpLower = builder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

              // Compare with upper bound
              Value splatUpperBound = builder.create<triton::SplatOp>(
                  loc, offsetWithRange.getType(), shape[i]);
              Value cmpUpper = builder.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::slt, offsetWithRange, splatUpperBound);

              // And and broadcast
              Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
              Value broadcasted =
                  builder.create<triton::BroadcastOp>(loc, maskTensorType, andResult);

              // And up all results
              if (!inited) {
                newMask = broadcasted;
                inited = true;
              } else {
                newMask = builder.create<arith::AndIOp>(loc, newMask, broadcasted);
              }
            }
          }

          // get other
          Value newOther;
          if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
            std::optional<triton::PaddingOption> padding = loadOp.getPadding();
            if (padding.has_value()) {
              // Create element attribute
              auto elementType = basePtrType.getPointeeType();
              auto otherTensorType = RankedTensorType::get(tensorShape, elementType);

              // Set zero padding value
              TypedAttr attr = builder.getZeroAttr(elementType);

              // Float NaN padding case
              if (padding.value() == triton::PaddingOption::PAD_NAN) {
                assert(!elementType.isIntOrIndex());
                auto apNaN = llvm::APFloat::getNaN(
                    cast<FloatAttr>(attr).getValue().getSemantics());
                attr = builder.getFloatAttr(elementType, apNaN);
              }

              // Create tensor
              Value constant = builder.create<arith::ConstantOp>(loc, attr);
              newOther = builder.create<triton::SplatOp>(loc, otherTensorType, constant);
            }
          }
         
          if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
            if (outlineTensorPtr) {
              auto newOp = rewriter.create<triton::LoadOp>(
                loadOp.getLoc(), newPtr, newMask, newOther, loadOp.getCache(),
                loadOp.getEvict(), loadOp.getIsVolatile());
              rewriter.replaceOp(op, newOp.getResult());
              
            } else {
              auto resultType = loadOp.getResult().getType();
              auto cacheAttr = triton::CacheModifierAttr::get(op->getContext(), loadOp.getCache());
              auto evictAttr = triton::EvictionPolicyAttr::get(op->getContext(), loadOp.getEvict());
              auto isVolatileAttr = rewriter.getBoolAttr(loadOp.getIsVolatile());
              
              auto newOp = rewriter.create<triton::ascend::UnstructuredLoadOp>(
                loc, resultType, base, ptr_offset,
                rewriter.getDenseI64ArrayAttr({}),
                newMask, newOther, cacheAttr, evictAttr, isVolatileAttr);

              rewriter.replaceOp(op, newOp.getResult());
            }
          } else  if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
            //@TODO
            auto newOp = rewriter.create<triton::StoreOp>(
              loc, newPtr, storeOp.getValue(), newMask,
              storeOp.getCache(), storeOp.getEvict());
            op->erase();
          }
        }
      }
    }
  }
  return success();
}

// Extract scalar compute from simt scope;
static LogicalResult extractScalarComputeFromSimtScope(ModuleOp moduleOp) {
  // 将scope中使用的外部常量复制一份到scope内部使用
  SmallVector<scope::ScopeOp> scopeOps;
  moduleOp.walk([&](scope::ScopeOp op) {
    if (auto vectorType = op->getAttrOfType<StringAttr>("vector_type")) {
      if (vectorType.str() != "simt") {
        return;
      }
    }
    scopeOps.push_back(op);
  });

  MLIRContext *context = moduleOp->getContext();
  IRRewriter rewriter(context);

  for (auto scopeOp : scopeOps) {
    // 收集所有在scope外部定义的常量
    DenseMap<Value, Value> externalConstants;
    
    scopeOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      for (auto &opr : op->getOpOperands()) {
        auto val = opr.get();
        // 检查是否是外部定义的常量
        if (auto *defOp = val.getDefiningOp()) {
          if (isa<arith::ConstantOp>(defOp) && !scopeOp->isAncestor(defOp)) {
            // 这是一个外部定义的常量
            if (externalConstants.find(val) == externalConstants.end()) {
              // 还没有复制过，需要复制
              OpBuilder builder(scopeOp.getRegion());
              builder.setInsertionPointToStart(&scopeOp.getRegion().front());
              auto newConstant = builder.create<arith::ConstantOp>(
                  defOp->getLoc(), cast<arith::ConstantOp>(defOp).getValue());
              externalConstants[val] = newConstant.getResult();
            }
          }
        }
      }
    });

    // 替换所有对外部常量的使用
    if (!externalConstants.empty()) {
      scopeOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
        for (auto &opr : op->getOpOperands()) {
          auto it = externalConstants.find(opr.get());
          if (it != externalConstants.end()) {
            opr.set(it->second);
          }
        }
      });
    }
  }
  
  return success();
}

// Outline simt scope to a function;
static LogicalResult outlineSimtScope(ModuleOp moduleOp) {
  SmallVector<scope::ScopeOp> scopeOps;
  moduleOp.walk([&](scope::ScopeOp op) {
    if (auto vectorType = op->getAttrOfType<StringAttr>("vector_type")) {
      if (vectorType.str() != "simt") {
        return;
      }
    }
    scopeOps.push_back(op);
  });

  MLIRContext *context = moduleOp->getContext();
  IRRewriter rewriter(context);

  static int scopeIndex = 0;
  for (auto scopeOp : scopeOps) {
    SmallVector<Value> inputs;
    scopeOp.walk<WalkOrder::PreOrder>([&inputs, &scopeOp](Operation *op) {
      for (auto &opr : op->getOpOperands()) {
        auto val = opr.get();
        if (auto blockArg = dyn_cast<BlockArgument>(val)) {
          if (scopeOp->isAncestor(blockArg.getParentRegion()->getParentOp()))
            continue;
        } else if (auto *defOp = val.getDefiningOp()) {
          if (scopeOp->isAncestor(defOp))
            continue;
        }
        inputs.insert(inputs.end(), val);
      }
    });

    llvm::sort(inputs, [](Value a, Value b) {
      if (a.getAsOpaquePointer() != b.getAsOpaquePointer())
        return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      return false;
    });
    inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());

    triton::FuncOp parentFunc = scopeOp->getParentOfType<triton::FuncOp>();
    std::string prefixFunctionName = parentFunc.getSymName().str() + "_simt_scope";

    rewriter.setInsertionPoint(parentFunc);
    FunctionType funcTy = FunctionType::get(
        moduleOp.getContext(), TypeRange(inputs), scopeOp->getResultTypes());
    func::FuncOp newFuncOp = rewriter.create<func::FuncOp>(
        moduleOp.getLoc(), prefixFunctionName, funcTy, scopeOp->getAttrs());
    SymbolTable symbolTable(moduleOp);
    FailureOr<StringAttr> scopeFuncName =
        symbolTable.renameToUnique(newFuncOp, SmallVector<SymbolTable *>());
    if (failed(scopeFuncName))
      return failure();

    Block *entryBB = newFuncOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBB);

    IRMapping currentMap;
    for (auto [oldIn, newIn] :
         llvm::zip_equal(inputs, entryBB->getArguments())) {
      currentMap.map(oldIn, newIn);
    }

    Operation *newScopeReturnOp = nullptr;
    for (auto it = scopeOp.getRegion().op_begin();
         it != scopeOp.getRegion().op_end(); ++it) {
      Operation *op = &*it;
      OpBuilder builder(op);
      auto *newOp = rewriter.clone(*op, currentMap);
      if (isa<scope::ReturnOp>(op))
        newScopeReturnOp = &*newOp;
    }
    if (!newScopeReturnOp)
      return failure();

    rewriter.create<func::ReturnOp>(
        entryBB->front().getLoc(), newScopeReturnOp->getOperands());
    rewriter.eraseOp(newScopeReturnOp);

    Block *scopeBlock = &scopeOp.getRegion().front();
    Operation *scopeReturnOp = scopeBlock->getTerminator();

    rewriter.setInsertionPointAfter(newFuncOp);
    auto privateVisibility = StringAttr::get(moduleOp.getContext(), "private");
    std::string uniqueDeclName = newFuncOp.getSymName().str() + "_decl";
    func::FuncOp newFuncDecl = rewriter.create<func::FuncOp>(
        newFuncOp.getLoc(), uniqueDeclName,
        newFuncOp.getFunctionType(), /*attrs=*/ArrayRef<NamedAttribute>{
            NamedAttribute("sym_visibility", privateVisibility)
        });

    rewriter.setInsertionPoint(scopeReturnOp);
    func::CallOp callOp = rewriter.create<func::CallOp>(
        scopeOp->getLoc(), newFuncDecl.getSymNameAttr(), scopeOp->getResultTypes(), inputs);

    static int fileIndex = 0;
    std::string fileName = "extracted_scope_" + std::to_string(fileIndex++) + ".mlir";
    std::error_code ec;
    llvm::raw_fd_ostream os(fileName, ec);
    if (!ec) {
      newFuncOp.print(os);
      os << "\n";
    }

    rewriter.replaceOp(newFuncOp, newFuncDecl);
  }

  return success();
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createVVMixPass() {
  return std::make_unique<VVMixPass>();
}

void VVMixPass::runOnOperation() {
  auto moduleOp = getOperation();

  if (failed(extractScalarComputeFromSimtScope(moduleOp))) {
    signalPassFailure();
    return;
  }

  if (failed(outlineSimtScope(moduleOp))) {
    signalPassFailure();
    return;
  }

  // if (failed(processTensorPtrInSimtScope(moduleOp))) {
  //   signalPassFailure();
  //   return;
  // }

  // @TODO: Experimental feature, to support more general mix-pipeline;
  if (0) {
    if (failed(outlineSimtScope(moduleOp))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace triton
} // namespace mlir