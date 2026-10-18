# profile

tree targets/reference, config None

10 iterations after a warmup of 10 (1 generator iteration(s) per 10); instrumented: the time is not comparable with `time`

window 37.60 s for 10 iterations; GPU busy 37.10 s (99%); kernel time 61.86 s (rank 0)

host time per iteration (s): 6.157, 3.498, 3.489, 3.502, 3.492, 3.491, 3.496, 3.489, 3.491, 3.495
kinds: gen, guidance, guidance, guidance, guidance, guidance, guidance, guidance, guidance, guidance

## kernel time by category (ms, rank 0)

| category | ms |
|---|---|
| other | 24018.7 |
| gemm | 17410.8 |
| attention | 10959.6 |
| memcpy/cast | 4001.6 |
| norm | 979.5 |
| nccl | 951.1 |
| elementwise/reduce | 906.3 |
| conv | 850.7 |
| optimizer | 427.0 |

## GPU time inside model calls (forward only; ranges nest: guidance_model contains real_unet, fake_unet, gan_head)

| call | GPU ms | calls | CPU ms |
|---|---|---|---|
| fake_unet | 4903.36 | 11 | 8139.58 |
| gan_head | 23.4 | 21 | 160.11 |
| generator | 9301.39 | 10 | 3016.03 |
| guidance_model | 8384.41 | 11 | 13257.61 |
| real_unet | 653.36 | 1 | 858.41 |
| text_encoder | 217.69 | 20 | 199.17 |

## top kernels

| ms | calls | share | kernel |
|---|---|---|---|
| 17773.344 | 21 | 0.2873 | `DistributedDataParallel.forward` |
| 4924.244 | 11 | 0.0796 | `## fake_unet` |
| 4280.979 | 100 | 0.0692 | `fmha_cutlassF_f32_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, true, true>::P` |
| 3238.655 | 194 | 0.0524 | `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<pytorch_flash::Flash_bwd_kernel_traits<64, 128, 128, 8, 4, 4, 4, false, fal` |
| 2192.229 | 314 | 0.0354 | `void pytorch_flash::flash_fwd_kernel<pytorch_flash::Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, cutlass::bfloat16_t, pytorch_flas` |
| 1699.202 | 431 | 0.0275 | `sm90_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x96x32_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_cudn` |
| 1693.553 | 4366 | 0.0274 | `sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas` |
| 1623.194 | 592 | 0.0262 | `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_c` |
| 1162.327 | 1500 | 0.0188 | `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x32_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas` |
| 1087.248 | 345 | 0.0176 | `sm90_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_c` |
| 1072.7 | 2974 | 0.0173 | `sm90_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas` |
| 925.698 | 260 | 0.015 | `sm90_xmma_fprop_implicit_gemm_f32f32_tf32f32_f32_nhwckrsc_nhwc_tilesize128x128x32_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_cud` |
| 920.726 | 327 | 0.0149 | `sm90_xmma_wgrad_indexed_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_warpgroupsize1x1x1_g1_execute_segment_k_on_kerne` |
| 877.801 | 21117 | 0.0142 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const:` |
| 836.735 | 760 | 0.0135 | `sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas` |
| 800.434 | 304 | 0.0129 | `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_warpgroupsize1x1x1_g1_execute_segment_k_on_kernel__5x_cu` |
| 739.58 | 15406 | 0.012 | `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const:` |
| 684.754 | 420 | 0.0111 | `sm90_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_c` |
| 655.813 | 1 | 0.0106 | `## real_unet` |
| 628.135 | 1140 | 0.0102 | `sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_on_kernel__5x_cublas` |
| 611.947 | 500 | 0.0099 | `sm90_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x32_warpgroupsize1x1x1_g1_execute_segment_k_off_kernel__5x_c` |
| 589.414 | 281 | 0.0095 | `sm90_xmma_wgrad_indexed_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x64x64_warpgroupsize1x1x1_g1_execute_segment_k_off_kerne` |
| 577.252 | 4788 | 0.0093 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIterat` |
| 532.667 | 100 | 0.0086 | `fmha_cutlassF_f32_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128, true, true>` |
| 496.891 | 7829 | 0.008 | `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, false, true, (cudnnKernelDataType_t)0>(cudnn::engines` |
| 477.242 | 811 | 0.0077 | `nccl:all_reduce` |
| 473.828 | 791 | 0.0077 | `ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevComm*, unsigned long, ncclWork*)` |
| 442.383 | 2477 | 0.0072 | `void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float, float>(int, float, float const*, float const*, float const*, flo` |
| 353.804 | 838 | 0.0057 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::` |
| 347.218 | 194 | 0.0056 | `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<pytorch_flash::Flash_bwd_kernel_traits<96, 64, 128, 8, 2, 4, 4, true, false` |
| 331.995 | 140 | 0.0054 | `sm90_xmma_dgrad_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_warpgroupsize1x1x1_g1_execute_segment_k_on_kernel__5x_cu` |
| 331.66 | 356 | 0.0054 | `sm90_xmma_wgrad_indexed_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nhwc_tilesize128x128x64_warpgroupsize1x1x1_g1_execute_segment_k_off_kern` |
| 304.031 | 190 | 0.0049 | `sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_on_kernel__5x_cublas` |
| 296.952 | 11 | 0.0048 | `Optimizer.step#AdamW.step` |
| 260.956 | 581 | 0.0042 | `sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas` |
| 260.75 | 1522 | 0.0042 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):` |
| 257.856 | 2331 | 0.0042 | `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::GroupNormKernelImplInterna` |
| 247.706 | 3577 | 0.004 | `void cudnn::engines_precompiled::nhwcToNchwKernel<__nv_bfloat16, __nv_bfloat16, float, true, false, (cudnnKernelDataType_t)0>(cudnn::engines` |
| 245.791 | 20 | 0.004 | `## text_encoder` |
| 242.794 | 515 | 0.0039 | `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, at::n` |

## top CPU ops (self time)

| ms | calls | op |
|---|---|---|
| 394.41 | 10771 | aten::mm |
| 237.16 | 47428 | aten::empty_strided |
| 233.52 | 5151 | aten::addmm |
| 230.99 | 56891 | aten::empty |
| 212.11 | 50587 | aten::copy_ |
| 163.65 | 1981 | aten::convolution_backward |
| 158.33 | 3444 | aten::cudnn_convolution |
| 102.47 | 36650 | aten::_to_copy |
| 101.63 | 27220 | aten::add_ |
| 90.9 | 2413 | aten::native_group_norm |
| 72.3 | 2525 | aten::native_layer_norm |
| 65.57 | 9972 | aten::mul |
| 61.69 | 58089 | aten::view |
| 60.81 | 4148 | aten::sum |
| 54.78 | 5313 | aten::add |
| 54.16 | 1280 | aten::native_group_norm_backward |
| 52.62 | 14938 | aten::linear |
| 52.38 | 16139 | aten::fill_ |
| 45.31 | 42988 | aten::transpose |
| 43.06 | 3675 | aten::convolution |
| 42.61 | 3262 | aten::div |
| 40.22 | 2882 | aten::silu |
| 39.66 | 70864 | aten::as_strided |
| 39.44 | 23250 | aten::t |
| 36.32 | 1030 | aten::_scaled_dot_product_flash_attention |
