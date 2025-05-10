---
title: Triton 入门-比 Torch 快的 OnlineSoftmax
tags: 学习笔记与作业
---

本文基于 Triton 逐步实现 online-softmax 算子，并与 Torch 的性能进行比较。实验结果显示，我的版本相较于 Torch 稳定快 26.1%。

$$
y_{i,j}=\frac{\exp (x_{i,j}-\max_j\lbrace x_{i,j}\rbrace)}{\max\lbrace\sum_j\exp (x_{i,j}-\max_j\lbrace x_{i,j}\rbrace),\epsilon\rbrace}
$$

本文只考虑二维张量下连续维度的 softmax。对于其他维度，转置（`rearrange`）后再连续维度 softmax 是更快的。

## 实验环境

- NVIDIA-A100-PCIE-40GB
- Debian 11
- spack@0.23.0
- cuda@12.6.2
- py-triton@2.1.0
- py-torch@2.4.1+cuda
- py-matplotlib@3.7.5
- py-pandas@1.5.3

## 源代码 `softmax.py`

分别实现了三个版本的 softmax：

1. softmax_fuse：每个 block 一次 load 一整列元素，直接算。这样做在 n 较小（`n_col < 65536`） 时 triton 的算子融合方式性能较好，否则性能差。推测过多的寄存器/SMEM占用影响了 occupancy。`n_col > 131072` 时直接无法启动。平均每个元素 1 次 load 1 次 store。
2. softmax_tile：按照列分tile处理寄存器/SMEM不够处理整列的问题。使用三次循环：第一次算整列的 max，第二次算整列的 sum of exp，第三次逐元素算 softmax。平均每个元素 3 次 load 1 次 store 2 次 exp（也可以改成 2 次 store 1 次 exp）。
3. softmax_online：在 softmax_tile 基础上将前两次 online 算 max 和 sum，这样可以减少 1 次 load 操作。由于 softmax 是访存密集算子，多几次 `exp` 交换也是可以接受的。

尤其值得注意的是 126 行处处理的 `nan` 问题，真的 debug 了很久。

顺带一提，我认为区间规约操作 `tl.max` `tl.sum` 开销很大，因此把他们都写在循环外面。

此外我还实现了一个 `CACHE_OPT` 策略，每次循环按照上次循环的相反方向进行，这样能尽可能 load 最近还停留在 cache 里的 block，这样做对 cache 更加友好。实测有效。

对 `num_warps=32`、`BLOCK_SIZE` 进行了一些手动调优。

```python
# spack load py-triton@2.1.0 py-torch@2.4.1+cuda py-matplotlib@3.7.5 py-pandas@1.5.3
# PATH=/usr/sbin:$PATH python3 softmax.py
import triton
import triton.language as tl
import torch


@triton.jit
def kernel_softmax_fuse(
    x_ptr,
    x_row_stride,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
    x = tl.exp(x - tl.max(x))
    eps = float(1e-9)
    x /= tl.maximum(tl.sum(x), eps)
    tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_fuse(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_fuse[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=32,
    )
    return y


@triton.jit
def kernel_softmax_tile(
    x_ptr,
    x_row_stride,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        mm = tl.maximum(mm, x)
    mm = tl.max(mm)

    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm)
            ss += x
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm)
            ss += x

    ss = tl.sum(ss)
    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        x = tl.exp(x - mm) / ss
        tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_tile(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_tile[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=2**14,
        CACHE_OPT=cache_opt,
        num_warps=32,
    )
    return y


@triton.jit
def kernel_softmax_online(
    x_ptr,
    x_row_stride,
    y_ptr,
    y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
        mm_new = tl.maximum(mm, x)
        if i:  # 第 1 轮不需要，且容易整出 nan
            ss *= tl.exp(mm - mm_new)
        x = tl.exp(x - mm_new)
        ss += tl.where(idx < n_cols, x, 0.0)
        mm = mm_new

    mm_new = tl.max(mm)
    ss *= tl.exp(mm - mm_new)
    ss = tl.sum(ss)
    mm = mm_new

    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask=idx < n_cols)
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask=idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask=idx < n_cols)


def triton_softmax_dim1_online(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_online[[n_rows]](
        x,
        x.stride(0),
        y,
        y.stride(0),
        n_cols,
        BLOCK_SIZE=2**12,
        CACHE_OPT=cache_opt,
        num_warps=32,
    )
    return y


def test():
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2**10, 2**15], device=DEVICE)
    mp = {
        "torch": lambda: torch.softmax(x, dim=1),
        "triton_fuse": lambda: triton_softmax_dim1_fuse(x),
        "triton_tile_no_cache": lambda: triton_softmax_dim1_tile(x, cache_opt=False),
        "triton_tile": lambda: triton_softmax_dim1_tile(x),
        "triton_online_no_cache": lambda: triton_softmax_dim1_online(
            x, cache_opt=False
        ),
        "triton_online": lambda: triton_softmax_dim1_online(x),
    }
    y_torch = mp["torch"]()
    for k, v in mp.items():
        y_triton = v()
        print("{}: Maxdiff is {}".format(k, torch.max(torch.abs(y_torch - y_triton))))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_col"],
        x_vals=[2**i for i in range(8, 18)],  # triton maximum tensor numel (131072)
        line_arg="provider",
        line_vals=[
            "torch",
            "triton_fuse",
            "triton_tile_no_cache",
            "triton_tile",
            "triton_online_no_cache",
            "triton_online",
        ],
        line_names=[
            "Torch",
            "Triton_fuse",
            "Triton_tile_no_cache",
            "Triton_tile",
            "Triton_online_no_cache",
            "Triton_online",
        ],
        plot_name="softmax-time",
        args={},
    )
)
def benchmark(n_col, provider):
    DEVICE = "cuda"  # triton.runtime.driver.active.get_active_torch_device()
    x = torch.rand([2**10, n_col], device=DEVICE)
    mp = {
        "torch": lambda: torch.softmax(x, dim=1),
        "triton_fuse": lambda: triton_softmax_dim1_fuse(x),
        "triton_tile_no_cache": lambda: triton_softmax_dim1_tile(x, cache_opt=False),
        "triton_tile": lambda: triton_softmax_dim1_tile(x),
        "triton_online_no_cache": lambda: triton_softmax_dim1_online(
            x, cache_opt=False
        ),
        "triton_online": lambda: triton_softmax_dim1_online(x),
    }
    return triton.testing.do_bench(mp[provider])  # ms


if __name__ == "__main__":
    torch.manual_seed(3407)
    test()
    benchmark.run(print_data=True, show_plots=False, save_path=".")
```

## 程序输出

```plain_text
torch: Maxdiff is 0.0
triton_fuse: Maxdiff is 1.0913936421275139e-11
triton_tile_no_cache: Maxdiff is 1.4551915228366852e-11
triton_tile: Maxdiff is 1.4551915228366852e-11
triton_online_no_cache: Maxdiff is 1.4551915228366852e-11
triton_online: Maxdiff is 1.4551915228366852e-11
softmax-time:
      n_col     Torch  Triton_fuse  Triton_tile_no_cache  Triton_tile  Triton_online_no_cache  Triton_online
0     256.0  0.008916     0.016236              0.062424     0.063158                0.026684       0.025777
1     512.0  0.011025     0.016656              0.062662     0.063190                0.027446       0.026267
2    1024.0  0.015170     0.017557              0.063223     0.063944                0.028480       0.027188
3    2048.0  0.024382     0.021070              0.064796     0.065755                0.029716       0.028612
4    4096.0  0.035157     0.032228              0.071570     0.072163                0.035416       0.034703
5    8192.0  0.061981     0.056555              0.079998     0.079716                0.061953       0.061838
6   16384.0  0.120682     0.106090              0.121865     0.122116                0.125531       0.122809
7   32768.0  0.362888     0.208714              0.232481     0.229461                0.297884       0.269076
8   65536.0  0.797070     0.630735              0.667213     0.597031                0.606105       0.569532
9  131072.0  1.569379     4.005666              1.566432     1.389664                1.196355       1.159837
```

![softmax-time](https://Mizuno-Ai.wu-kan.cn/assets/image/2024/11/21/softmax-time.png)
