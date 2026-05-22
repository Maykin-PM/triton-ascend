#include "./RegisterTritonDialects.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/VVMix/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  // Register AddMultiBufferInnerScope pass
  mlir::triton::registerAddMultiBufferInnerScopePasses();
  // Register VVMix pass
  mlir::triton::registerVVMix();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
