---
title: Triton 入门-Relu-并导出中间表示
tags: 学习笔记与作业
---

学习一下 triton。作为本系列文章的第一篇内容，我希望写出一段尽可能简短易懂的代码，同时能够导出从 triton kernel 到最终 LLVM-IR/PTX 后端表示过程中的一系列变换。最终只用了四十五行就完成了任务，足以见 triton 设计上的简洁。

## 实验环境

- NVIDIA-A100-PCIE-40GB
- Debian 11
- spack@0.23.0
- cuda@12.6.2
- py-triton@2.1.0
- py-torch@2.4.1+cuda

## 源代码 `relu.py`

配置环境的时候遇到一个问题，运行时需要用的 `ldconfig` 在 `/usr/sbin/` 目录下，很奇怪为啥要这么搞。我的解决方式是在运行时加入 `PATH`。

以下是首次上手编写的 `relu` 函数，作为一个最小化的上手例子。同时导出运行时的层层下降的中间表示 `['ast', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']`，用于观察 triton 生成算子的具体逻辑。

```python
# spack load py-triton@2.1.0 py-torch@2.4.1+cuda
# PATH=/usr/sbin:$PATH python3 relu.py
import triton
import triton.language as tl
import torch


@triton.jit
def kernel_relu(x_ptr, y_ptr, size, BLOCK_SIZE: tl.constexpr):
    # tl.arange 返回一个数组 [0, 1 , ..., BLOCK_SIZE - 1]
    # tl.program_id(0) * BLOCK_SIZE 是一个数，加在一起 idx 是一个数组
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask 和 idx 一样是数组
    mask = idx < size
    # x_ptr + idx 是数组，所以可以一次 load 一整个数组
    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.maximum(x, 0)
    tl.store(y_ptr + idx, y, mask=mask)


def triton_relu(x: torch.Tensor):
    assert x.is_contiguous()  # 确保输入是1D张量
    size = x.numel()
    y = torch.empty_like(x)
    gridDim = lambda meta: [triton.cdiv(size, meta["BLOCK_SIZE"])]
    kernel_relu[gridDim](x, y, size, BLOCK_SIZE=256)
    return y


if __name__ == "__main__":
    torch.manual_seed(3407)
    compiled_relu = triton.compile(
        kernel_relu, signature="*fp32,*fp32,i32", constants={"BLOCK_SIZE": 256}
    )
    for k, v in compiled_relu.asm.items():
        with open("relu." + k, "wb" if type(v) == bytes else "w") as f:
            f.write(v)
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand(2**20, device=DEVICE)
    x -= 0.5
    y_torch = torch.relu(x)
    y_triton = triton_relu(x)
    print(y_torch)
    print(y_triton)
    print("Maxdiff is {}".format(torch.max(torch.abs(y_torch - y_triton))))
```

## 程序输出

```plain_text
tensor([0.3659, 0.0000, 0.0000,  ..., 0.0000, 0.3017, 0.0000], device='cuda:0')
tensor([0.3659, 0.0000, 0.0000,  ..., 0.0000, 0.3017, 0.0000], device='cuda:0')
Maxdiff is 0.0
```

## `relu.ast`

很奇怪为啥语法树只输出了这么多。

```plain_text
JITFunction(__main__:relu_kernel)
```

## `relu.ttir`

标准的 mlir，还是比较好读的。

```plain_text
module {
  tt.func public @kernel_relu_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32>
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg2 : (i32) -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
    %10 = arith.cmpf ogt, %9, %cst : tensor<256xf32>
    %11 = arith.select %10, %9, %cst : tensor<256xi1>, tensor<256xf32>
    %12 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %13, %11, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<256xf32>
    tt.return
  }
}
```

## `relu.ttgir`

还是 mlir，tt gpu ir。与 `relu.ttir` 相比，主要是多了 `#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>` （同时标记在每个 tensor 上）以及 `attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32}`，硬件相关的计算表示。

```plain_text
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_relu_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked>
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked>
    %5 = tt.splat %arg2 : (i32) -> tensor<256xi32, #blocked>
    %6 = "triton_gpu.cmpi"(%4, %5) <{predicate = 2 : i64}> : (tensor<256xi32, #blocked>, tensor<256xi32, #blocked>) -> tensor<256xi1, #blocked>
    %7 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked>
    %10 = "triton_gpu.cmpf"(%9, %cst) <{predicate = 2 : i64}> : (tensor<256xf32, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xi1, #blocked>
    %11 = "triton_gpu.select"(%10, %9, %cst) : (tensor<256xi1, #blocked>, tensor<256xf32, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    %12 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %13, %11, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<256xf32, #blocked>
    tt.return
  }
}
```

## `relu.llir`

llvmir，没啥好说的。

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @kernel_relu_012(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2) local_unnamed_addr !dbg !5 {
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !8
  %5 = and i32 %4, 127, !dbg !8
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !9
  %7 = shl i32 %6, 8, !dbg !10
  %8 = or i32 %7, %5, !dbg !11
  %9 = or i32 %8, 128, !dbg !11
  %10 = icmp slt i32 %8, %2, !dbg !12
  %11 = icmp slt i32 %9, %2, !dbg !12
  %12 = sext i32 %8 to i64, !dbg !13
  %13 = getelementptr float, ptr addrspace(1) %0, i64 %12, !dbg !13
  %14 = sext i32 %9 to i64, !dbg !13
  %15 = getelementptr float, ptr addrspace(1) %0, i64 %14, !dbg !13
  %16 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b"(ptr addrspace(1) %13, i1 %10) #1, !dbg !14
  %17 = bitcast i32 %16 to float, !dbg !14
  %18 = tail call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09@$2 ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b"(ptr addrspace(1) %15, i1 %11) #1, !dbg !14
  %19 = bitcast i32 %18 to float, !dbg !14
  %20 = fcmp ogt float %17, 0.000000e+00, !dbg !15
  %21 = fcmp ogt float %19, 0.000000e+00, !dbg !15
  %22 = select i1 %20, float %17, float 0.000000e+00, !dbg !19
  %23 = select i1 %21, float %19, float 0.000000e+00, !dbg !19
  %24 = getelementptr float, ptr addrspace(1) %1, i64 %12, !dbg !20
  %25 = getelementptr float, ptr addrspace(1) %1, i64 %14, !dbg !20
  %26 = bitcast float %22 to i32, !dbg !21
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %26, ptr addrspace(1) %24, i1 %10) #1, !dbg !21
  %27 = bitcast float %23 to i32, !dbg !21
  tail call void asm sideeffect "@$2 st.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"(i32 %27, ptr addrspace(1) %25, i1 %11) #1, !dbg !21
  ret void, !dbg !22
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!nvvm.annotations = !{!3, !4, !4, !3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "relu.py", directory: "")
!3 = !{ptr @kernel_relu_012, !"kernel", i32 1}
!4 = !{ptr @kernel_relu_012, !"maxntidx", i32 128}
!5 = distinct !DISubprogram(name: "kernel_relu_012", linkageName: "kernel_relu_012", scope: !2, file: !2, line: 9, type: !6, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1)
!6 = !DISubroutineType(cc: DW_CC_normal, types: !7)
!7 = !{}
!8 = !DILocation(line: 12, column: 55, scope: !5)
!9 = !DILocation(line: 12, column: 24, scope: !5)
!10 = !DILocation(line: 12, column: 29, scope: !5)
!11 = !DILocation(line: 12, column: 42, scope: !5)
!12 = !DILocation(line: 14, column: 17, scope: !5)
!13 = !DILocation(line: 16, column: 24, scope: !5)
!14 = !DILocation(line: 16, column: 16, scope: !5)
!15 = !DILocation(line: 1398, column: 21, scope: !16, inlinedAt: !18)
!16 = distinct !DILexicalBlockFile(scope: !5, file: !17, discriminator: 0)
!17 = !DIFile(filename: "core.py", directory: "/mnt/sda/2022-0526/public/wuk/v0.23.0.12.3.0.12.20241120/spack/opt/spack/linux-debian11-zen2/gcc-13.2.0/py-triton-2.1.0-564r5tjbwbaxjbq6av25ende3vpeukfk/lib/python3.8/site-packages/triton/language")
!18 = !DILocation(line: 17, column: 22, scope: !16)
!19 = !DILocation(line: 1398, column: 27, scope: !16, inlinedAt: !18)
!20 = !DILocation(line: 18, column: 21, scope: !5)
!21 = !DILocation(line: 18, column: 26, scope: !5)
!22 = !DILocation(line: 18, column: 4, scope: !5)
```

## `relu.ptx`

生成最终的 PTX。注意到针对最终机器设置了 `.target sm_80`。LLVM 的 PTX backend 版本落后于 nvcc，性能可能会有所影响。

```plain_text
//
// Generated by LLVM NVPTX Back-End
//

.version 8.1
.target sm_80
.address_size 64

	// .globl	kernel_relu_012

.visible .entry kernel_relu_012(
	.param .u64 kernel_relu_012_param_0,
	.param .u64 kernel_relu_012_param_1,
	.param .u32 kernel_relu_012_param_2
)
.maxntid 128, 1, 1
{
	.reg .pred 	%p<5>;
	.reg .b32 	%r<12>;
	.reg .f32 	%f<5>;
	.reg .b64 	%rd<8>;
	.loc	1 9 0
$L__func_begin0:
	.loc	1 9 0

	ld.param.u64 	%rd5, [kernel_relu_012_param_0];
	ld.param.u64 	%rd6, [kernel_relu_012_param_1];
$L__tmp0:
	.loc	1 12 55
	mov.u32 	%r5, %tid.x;
	and.b32  	%r6, %r5, 127;
	ld.param.u32 	%r7, [kernel_relu_012_param_2];
	.loc	1 12 24
	mov.u32 	%r8, %ctaid.x;
	.loc	1 12 29
	shl.b32 	%r9, %r8, 8;
	.loc	1 12 42
	or.b32  	%r10, %r9, %r6;
	or.b32  	%r11, %r10, 128;
	.loc	1 14 17
	setp.lt.s32 	%p1, %r10, %r7;
	setp.lt.s32 	%p2, %r11, %r7;
	.loc	1 16 24
	mul.wide.s32 	%rd7, %r10, 4;
	add.s64 	%rd1, %rd5, %rd7;
	add.s64 	%rd2, %rd1, 512;
	.loc	1 16 16
	mov.u32 %r1, 0x0;
	@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
	mov.b32 	%f1, %r1;
	mov.u32 %r2, 0x0;
	@%p2 ld.global.b32 { %r2 }, [ %rd2 + 0 ];
	mov.b32 	%f2, %r2;
$L__tmp1:
	.loc	2 1398 27
	max.f32 	%f3, %f1, 0f00000000;
	max.f32 	%f4, %f2, 0f00000000;
$L__tmp2:
	.loc	1 18 21
	add.s64 	%rd3, %rd6, %rd7;
	add.s64 	%rd4, %rd3, 512;
	.loc	1 18 26
	mov.b32 	%r3, %f3;
	@%p1 st.global.b32 [ %rd3 + 0 ], { %r3 };
	mov.b32 	%r4, %f4;
	@%p2 st.global.b32 [ %rd4 + 0 ], { %r4 };
	.loc	1 18 4
	ret;
$L__tmp3:
$L__func_end0:

}
	.file	1 "relu.py"
	.file	2 "/mnt/sda/2022-0526/public/wuk/v0.23.0.12.3.0.12.20241120/spack/opt/spack/linux-debian11-zen2/gcc-13.2.0/py-triton-2.1.0-564r5tjbwbaxjbq6av25ende3vpeukfk/lib/python3.8/site-packages/triton/language/core.py"
	.section	.debug_abbrev
	{
.b8 1
.b8 17
.b8 1
.b8 37
.b8 8
.b8 19
.b8 5
.b8 3
.b8 8
.b8 16
.b8 6
.b8 180
.b8 66
.b8 12
.b8 17
.b8 1
.b8 18
.b8 1
.b8 0
.b8 0
.b8 2
.b8 46
.b8 0
.b8 135
.b8 64
.b8 8
.b8 3
.b8 8
.b8 58
.b8 11
.b8 59
.b8 11
.b8 63
.b8 12
.b8 32
.b8 11
.b8 0
.b8 0
.b8 3
.b8 46
.b8 1
.b8 17
.b8 1
.b8 18
.b8 1
.b8 64
.b8 10
.b8 49
.b8 19
.b8 0
.b8 0
.b8 4
.b8 29
.b8 0
.b8 49
.b8 19
.b8 17
.b8 1
.b8 18
.b8 1
.b8 88
.b8 11
.b8 89
.b8 11
.b8 87
.b8 11
.b8 0
.b8 0
.b8 0
	}
	.section	.debug_info
	{
.b32 132
.b8 2
.b8 0
.b32 .debug_abbrev
.b8 8
.b8 1
.b8 116
.b8 114
.b8 105
.b8 116
.b8 111
.b8 110
.b8 0
.b8 2
.b8 0
.b8 114
.b8 101
.b8 108
.b8 117
.b8 46
.b8 112
.b8 121
.b8 0
.b32 .debug_line
.b8 1
.b64 $L__func_begin0
.b64 $L__func_end0
.b8 2
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 95
.b8 114
.b8 101
.b8 108
.b8 117
.b8 95
.b8 48
.b8 49
.b8 50
.b8 0
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 95
.b8 114
.b8 101
.b8 108
.b8 117
.b8 95
.b8 48
.b8 49
.b8 50
.b8 0
.b8 1
.b8 9
.b8 1
.b8 1
.b8 3
.b64 $L__func_begin0
.b64 $L__func_end0
.b8 1
.b8 156
.b32 50
.b8 4
.b32 50
.b64 $L__tmp1
.b64 $L__tmp2
.b8 2
.b8 17
.b8 22
.b8 0
.b8 0
	}
	.section	.debug_pubnames
	{
.b32 $L__pubNames_end0-$L__pubNames_start0
$L__pubNames_start0:
.b8 2
.b8 0
.b32 .debug_info
.b32 136
.b32 50
.b8 107
.b8 101
.b8 114
.b8 110
.b8 101
.b8 108
.b8 95
.b8 114
.b8 101
.b8 108
.b8 117
.b8 95
.b8 48
.b8 49
.b8 50
.b8 0
.b32 0
$L__pubNames_end0:
	}
	.section	.debug_pubtypes
	{
.b32 $L__pubTypes_end0-$L__pubTypes_start0
$L__pubTypes_start0:
.b8 2
.b8 0
.b32 .debug_info
.b32 136
.b32 0
$L__pubTypes_end0:
	}
	.section	.debug_loc	{	}
```
