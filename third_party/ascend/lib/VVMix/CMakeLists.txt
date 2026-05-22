add_triton_library(VVMix
        VVMixPass.cpp

  DEPENDS
  VVMixPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
  MLIRFuncDialect
)