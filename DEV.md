* 出包：
TRITON_PLUGIN_DIRS=./ascend \
TRITON_BUILD_WITH_CCACHE=true TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_PROTON=OFF TRITON_WHEEL_NAME="triton-ascend" TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" python3 setup.py bdist_wheel

* 构建：
# 1. 进入构建目录
cd ./build/cmake.linux-aarch64-cpython-3.13

# 2. 配置 CMake（Debug 模式，禁用 strip）
cmake -S ../../../triton-ascend -B . \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -Wl,-z,now -pie -Wl,--gc-sections"

# 3. 移除 LINK_FLAGS 中的 -s 标志（防止符号剥离）
sed -i 's/-Wl,-z,now -pie -s/-Wl,-z,now -pie/g' build.ninja

# 4. 清理并重新编译 triton-opt
ninja -t clean && ninja -j$(nproc) triton-opt

* pass 调试
bin/triton-opt kernel.mlir --auto-blockify="auto-blockify-size=1" --triton-to-structured --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg  --mlir-print-ir-after-all --mlir-print-debuginfo &> 0.log

bin/triton-opt debug.mlir --vv-mix --auto-blockify="auto-blockify-size=1" --triton-to-structured --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation --triton-to-structured --triton-to-linalg  --mlir-print-ir-after-all --mlir-print-debuginfo &> 0.log

* IR:
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func private @foo(!tt.ptr<f32>) -> (tensor<16x16xf32>)
  tt.func public @merge_16x16_to_64x64_inverse_kernel_mix(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32) attributes {noinline = false} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.remsi %1, %c32_i32 : i32
    %4 = arith.muli %2, %arg2 : i32
    %5:2 = scope.scope : () -> (!tt.ptr<f32>, tensor<16x16xf32>) {
      %10 = arith.muli %4, %c32_i32 : i32
      %11 = arith.addi %10, %3 : i32
      %12 = arith.muli %11, %c64_i32 : i32
      %13 = tt.addptr %arg0, %12 : !tt.ptr<f32>, i32
      %14 = arith.muli %0, %c64_i32 : i32
      %15 = arith.extsi %arg2 : i32 to i64
      %16 = tt.make_tensor_ptr %13, [%15, %c64_i64], [%c2048_i64, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>
      %17 = tt.load %16 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x16xf32>>
      %18 = func.call @foo(%13) : (!tt.ptr<f32>) -> tensor<16x16xf32>
      scope.return %13, %17 : !tt.ptr<f32>, tensor<16x16xf32>
    } {noinline, vector_type = "simt"}
    %6 = arith.muli %0, %c64_i32 : i32
    %7 = arith.addi %6, %c16_i32 : i32
    %8 = arith.extsi %arg2 : i32 to i64
    //%9 = tt.make_tensor_ptr %arg0, [%8, %c64_i64], [%c2048_i64, %c1_i64], [%7, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>

    %22 = tt.addptr %5#0, %c32_i32 : !tt.ptr<f32>, i32
    %9 = tt.make_tensor_ptr %22, [%8, %c64_i64], [%c2048_i64, %c1_i64], [%7, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>>

    tt.store %9, %5#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x16xf32>>
    tt.return
  }
}