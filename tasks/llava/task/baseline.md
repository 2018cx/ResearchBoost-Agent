# Baseline (reference, 4 × H20, upstream recipe unchanged)

- Workload: 16 per GPU × gradient accumulation 2 × 4 GPUs = global 128, model_max_length 2048, bf16 + tf32, ZeRO-3, flash-attn 2, gradient checkpointing
- Timing plan: 2 warmup steps, 2 windows of 8 steps each
- **20960 ms per step** (two reference runs 20968 / 20952 ms, relative gap 0.08%)
- Throughput 6.10 samples/s (all 4 GPUs)
- Peak memory 47.1 GiB allocated / 66.4 GiB reserved (largest GPU)
- 46 s from startup to the first step (loading the 7B weights into ZeRO-3 + CLIP + projector)
- Median GPU utilization inside the windows 100.0%, average power of the 4 GPUs 1370.4 W
- Windows of the two reference runs (s): [169.469, 166.022] / [169.294, 165.932]

## Profile summary (profile_reference.md)

## cuda_gpu_kern_sum

    Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
    29.1,62245818787,6688,9307090.1,6584010.0,2182628,43973321,4823649.1,sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas
    14.9,31752603495,9608,3304808.9,438573.5,6944,86601852,4944087.8,"ncclKernel_AllGather_RING_LL_Sum_int8_t(ncclDevComm *, unsigned long, ncclWork *)"
    14.1,30108424368,3088,9750137.4,7618589.0,2182884,29779499,5016576.1,sm90_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas
    11.3,24179472437,4688,5157737.3,2925022.5,7936,98288109,6519920.6,"ncclKernel_ReduceScatter_RING_LL_Sum___nv_bfloat16(ncclDevComm *, unsigned long, ncclWork *)"
    9.7,20683309693,1552,13326874.8,12617324.0,11331731,56462143,3403146.0,sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas
    5.3,11348210885,2080,5455870.6,5129492.0,661633,8807693,1125521.7,sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas
    1.8,3887546931,9248,420366.2,411646.0,308320,1053988,95028.8,"void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_i
    1.5,3211478938,528,6082346.5,4418770.0,4374066,54666478,6775258.1,sm90_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas
    1.3,2672794728,896,2983029.8,4392578.0,591809,11266909,2017649.5,sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas
    1.0,2207162829,512,4310864.9,3850986.0,3411414,8838497,965130.0,"void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<(int)128, (int)64, (int)128, (int)8, (int)2, (int)4, (int)2, (bool)0, (bool)0, cutlass::bfloat16_t, F
    0.7,1600150289,5424,295013.0,397567.0,34976,1110148,211891.8,"void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<c10::BFloat16>>(at::TensorIteratorBase &, const T1 &)::[lambda(
    0.7,1596760929,3192,500238.4,479776.0,2624,2487288,241144.3,"void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 12)]:
    0.7,1587631930,3176,499884.1,478030.0,2848,2389674,205453.0,"void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 7)]::
    0.7,1537222771,1024,1501194.1,1487591.5,1416706,3830739,123565.3,"void flash_fwd_kernel<Flash_fwd_kernel_traits<(int)128, (int)128, (int)64, (int)4, (bool)0, (bool)0, cutlass::bfloat16_t, Flash_kernel_traits<(int)128, (int)128, (int)64, (in
    0.7,1504253533,3104,484617.8,490223.5,354399,1021338,85986.2,"void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<floa
    0.5,1146163408,2048,559650.1,527873.0,490719,1464454,86362.4,"void at::native::<unnamed>::CatArrayBatchedCopy<c10::BFloat16, unsigned int, (int)4, (int)64, (int)64>(T1 *, at::native::<unnamed>::CatArrInputTensorMetadata<T1, T2, T4, T5>, at:
    0.5,1127052690,3472,324611.9,373374.0,73855,623517,155691.8,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>, at::
    0.5,1117835465,9240,120977.9,143071.0,1472,601342,73159.4,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<c10::BFloat16>, at::detail::Array<char *, (int)3>>(int, T2, T3)"
    0.4,953895615,4656,204874.5,191712.0,3392,875297,140764.6,"void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operat
    0.4,895281725,3072,291432.9,285729.0,276224,640094,29110.6,"void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)0, at::native::OpaqueType<(int)2>>::operator ()<at
    0.3,692507531,1024,676276.9,672499.0,476350,1086913,127756.8,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::silu_kernel(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda() (inst
    0.3,570692905,3120,182914.4,255263.5,1312,409918,130235.2,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::detail::Array<char *, (int
    0.3,549802891,1536,357944.6,378303.0,276064,640254,80815.8,"void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)4, void at::native::_cuda_scatter_gather_internal_kernel<(bool)1, at::native::OpaqueType<(int)2>>::operator ()<at
    0.3,549762139,3072,178959.0,166464.0,149024,429889,37794.9,"void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl<at::native::neg_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()()
    0.3,544329605,1936,281162.0,227200.0,151201,825147,199782.8,sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas
    0.2,511180075,1024,499199.3,435455.0,420255,1090652,90124.9,"void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_put_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIter
    0.2,496464942,5615,88417.6,88960.0,1056,155393,32314.6,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::BFloat16>, at::detail::Array<char *, (int)1>>(int, T2, T3)"
    0.2,421907991,2064,204412.8,195009.0,174367,354526,22644.7,"void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase &, T2)::[lambda(float) (instan
    0.1,319902821,1040,307598.9,314320.5,232544,682205,54643.7,"void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>
    0.1,317862273,512,620824.8,578413.0,501503,921084,97746.1,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::<unnamed>::silu_backward_kernel(at::TensorIteratorBase &)::[lambda() (instance 1)]::operator ()() const::[lambda()
    0.1,280315320,1064,263454.2,257631.0,992,407134,48493.8,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)"
    0.1,264419180,2560,103288.7,5760.0,4896,632284,199108.6,"void at::native::index_elementwise_kernel<(int)128, (int)4, void at::native::gpu_index_kernel<void at::native::index_kernel_impl<at::native::OpaqueType<(int)2>>(at::TensorIteratorBase
    0.1,248171770,2088,118856.2,119169.0,2144,210623,17916.1,"void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, (int)4>>(T3)"
    0.1,242418972,2152,112648.2,4608.0,1184,1181402,149198.3,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::detail::Array<char *, (int)
    0.1,206690782,384,538257.2,474222.0,465472,683582,90209.4,void cutlass::Kernel<cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1>(T1::Params)
    0.1,195842764,4688,41775.3,35136.0,1248,311840,32983.0,"void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float>>, at::detai
    0.1,166097650,384,432546.0,381567.5,370465,541918,71185.5,void cutlass::Kernel<cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_nn_align1>(T1::Params)
    0.1,129300183,384,336719.2,289872.0,287392,436993,57657.5,"void <unnamed>::softmax_warp_forward<c10::BFloat16, c10::BFloat16, float, (int)10, (bool)0, (bool)0>(T2 *, const T1 *, int, int, int, const bool *, int, bool)"
    0.1,122718031,1048,117097.4,118080.5,2177,210271,16302.3,"void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator ()(at::TensorIte
    0.1,112860028,64,1763437.9,1836369.0,1354908,2482807,327974.6,"void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::op

## cuda_gpu_mem_time_sum

    Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation
    94.4,653947999,8744,74788.2,21631.5,896,3990130,182060.0,[CUDA memcpy Device-to-Device]
    3.1,21511139,19680,1093.0,1056.0,735,16000,486.8,[CUDA memset]
    1.5,10341487,4200,2462.3,2432.0,2112,3968,199.4,[CUDA memcpy Device-to-Host]
    1.0,7152827,72,99344.8,4352.0,800,482304,181880.9,[CUDA memcpy Host-to-Device]

## nvtx_sum

    Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Style,Range
    25.7,116983165905,16,7311447869.1,7119387036.0,6496706666,8536652609,761759925.5,PushPop,:DeepSpeedEngine.backward
    25.3,115270206609,16,7204387913.1,7014135102.5,6394730503,8425991128,753492809.0,PushPop,:DeepSpeedZeroOptimizer_Stage3.backward
    17.4,79403990181,4720,16822879.3,5959622.0,59219,103591295,23837055.2,PushPop,:DeepSpeedZeroOptimizer_Stage3.create_reduce_and_remove_grad_hooks.<locals>.wrapper.<locals>.reduce_partition_and_remove_grads
    17.3,78997231542,4688,16850945.3,6708900.5,320985,103492362,23861317.6,PushPop,:DeepSpeedZeroOptimizer_Stage3.__reduce_and_partition_ipg_grads
    9.1,41498512147,16,2593657009.2,2527145021.0,2381737068,2970256999,217913168.6,PushPop,:DeepSpeedEngine.forward
    0.8,3552934723,26512,134012.3,100998.0,28240,2165130,116049.9,PushPop,:PartitionedParameterCoordinator.fetch_sub_module
    0.6,2809549561,19472,144286.6,111930.0,38019,2180125,120192.3,PushPop,:DeepSpeedZeRoOffload._register_hooks_recursively.<locals>._pre_forward_module_hook
    0.3,1332976433,25680,51907.2,19676.5,10217,958140,49567.2,PushPop,:PartitionedParameterCoordinator.release_sub_module
    0.3,1312146322,7096,184913.5,179724.0,107123,2077924,61645.3,PushPop,:PartitionedParameterCoordinator.__all_gather_params
    0.3,1201057079,4688,256198.2,247586.5,179316,2881589,74380.7,PushPop,:DeepSpeedZeroOptimizer_Stage3.__avg_scatter_grads
    0.2,1124131263,19472,57730.7,31427.5,18246,971369,48619.6,PushPop,:DeepSpeedZeRoOffload._register_hooks_recursively.<locals>._post_forward_module_hook
    0.2,1118500711,4688,238588.0,230613.5,168898,2386024,68673.0,PushPop,:reduce_scatter_coalesced
    0.2,1082948867,7096,152614.0,147820.0,83289,2036101,56975.6,PushPop,:Init._convert_to_deepspeed_param.<locals>.all_gather_coalesced
    0.2,1044047238,7040,148302.2,75317.0,36930,2236730,111351.1,PushPop,:DeepSpeedZeRoOffload._register_hooks_recursively.<locals>._pre_backward_module_hook.<locals>._run_before_backward_function
    0.2,989369448,8,123671181.0,123517115.0,122867906,125327426,890450.9,PushPop,:DeepSpeedZeroOptimizer_Stage3.step
    0.2,923419939,14640,63075.1,77309.5,828,931509,40631.5,PushPop,:PartitionedParameterCoordinator.__release_param
    0.1,667290673,7096,94037.6,90852.5,52790,1955451,47125.1,PushPop,:allgather_fn
    0.1,614753556,11648,52777.6,53244.5,25572,896132,20499.4,PushPop,:Init._partition_param
    0.1,613099361,4688,130780.6,126239.5,92876,1530082,47261.1,PushPop,:reduce_scatter_fn
    0.1,541055479,8,67631934.9,67605237.5,67245849,68204188,309651.4,PushPop,:DeepSpeedZeroOptimizer_Stage3._post_step
    0.1,540702828,8,67587853.5,67576142.5,67210658,68092753,291077.0,PushPop,:Init._all_gather
    0.1,539503275,6208,86904.5,113463.0,18437,975390,51384.5,PushPop,:DeepSpeedZeRoOffload._register_hooks_recursively.<locals>._post_backward_module_hook.<locals>._run_after_backward_function
    0.1,423946460,11648,36396.5,35896.0,17073,877116,17131.9,PushPop,:free_param
    0.1,396714880,14320,27703.6,27581.0,8592,651062,13491.5,PushPop,NCCL:ncclGroupEnd
    0.1,339284532,4720,71882.3,80409.5,23934,2346172,48659.9,PushPop,:DeepSpeedZeroOptimizer_Stage3.__add_grad_to_ipg_bucket
    0.1,330585400,4688,70517.4,67346.5,49006,821513,22098.8,PushPop,:DeepSpeedZeroOptimizer_Stage3.partition_grads
    0.1,276637706,16,17289856.6,17300996.0,16686076,17860638,392234.0,PushPop,:DeepSpeedEngine.allreduce_gradients
    0.1,276308985,16,17269311.6,17283793.0,16665661,17833192,390358.2,PushPop,:DeepSpeedZeroOptimizer_Stage3.independent_gradient_partition_epilogue
    0.0,215491733,8,26936466.6,26730967.0,26016332,28135471,797607.9,PushPop,:DeepSpeedZeroOptimizer_Stage3._get_norm_groups
    0.0,215308361,24,8971181.7,9471312.5,1971924,16699364,5642862.0,PushPop,:DeepSpeedZeroOptimizer_Stage3.get_grad_norm_direct
    0.0,168722292,2352,71735.7,1477.0,936,1322349,188154.4,PushPop,:AllGatherCoalescedHandle.wait
    0.0,164690334,8,20586291.8,20286420.0,19928245,21939357,777412.3,PushPop,:DeepSpeedZeroOptimizer_Stage3._partition_all_parameters
    0.0,164630471,8,20578808.9,20280593.0,19922709,21931058,775469.5,PushPop,:DeepSpeedZeRoOffload.partition_all_parameters
    0.0,154610375,8,19326296.9,18989254.5,18716598,20651971,749823.8,PushPop,:PartitionedParameterCoordinator.release_and_reset_all
    0.0,138824583,52216,2658.7,2326.0,1258,290467,2086.3,PushPop,:get_all_parameters
    0.0,85437746,7096,12040.3,11235.5,6075,311888,5923.7,PushPop,:PyCapsule.wait
    0.0,60861563,25680,2370.0,2132.0,939,127009,1783.0,PushPop,:PartitionedParameterCoordinator.__params_to_release
    0.0,55354570,2664,20778.7,16187.0,10223,271249,12085.8,PushPop,:_VariableFunctionsClass.cat
    0.0,27074749,9608,2817.9,2997.0,684,312082,3666.4,PushPop,NCCL:ncclAllGather
    0.0,23926471,24,996936.3,915098.0,674470,1425970,296625.4,PushPop,:DeepSpeedZeroOptimizer_Stage3._prepare_sub_group

## Profile summary (profile_reference_detail.md)

# detail profile, rank 0, 1 step(s)

GPU kernel time: 25812 ms/step

| kernel | ms/step | share |
|---|---|---|
| sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas | 8908.6 | 34.5% |
| ncclKernel_AllGather_RING_LL_Sum_int8_t(ncclDevComm*, unsigned long, ncclWork*) | 3740.8 | 14.5% |
| sm90_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas | 3455.6 | 13.4% |
| sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas | 2913.8 | 11.3% |
| sm90_xmma_gemm_bf16bf16_bf16f32_f32_nt_n_tilesize128x256x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas | 1476.1 | 5.7% |
| ncclKernel_ReduceScatter_RING_LL_Sum___nv_bfloat16(ncclDevComm*, unsigned long, ncclWork*) | 848.7 | 3.3% |
| sm90_xmma_gemm_bf16bf16_bf16f32_f32_nn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas | 840.3 | 3.3% |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binar | 516.3 | 2.0% |
| void flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 128, 8, 2, 4, 2, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128,  | 268.9 | 1.0% |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#12}::oper | 215.3 | 0.8% |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<c10::BFloat16> >(at::TensorIteratorBase&, at::native::CUDAFu | 212.7 | 0.8% |
| void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#7}::opera | 208.2 | 0.8% |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<f | 202.3 | 0.8% |
| void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, f | 198.3 | 0.8% |
| void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor | 158.1 | 0.6% |
| void at::native::(anonymous namespace)::CatArrayBatchedCopy<c10::BFloat16, unsigned int, 4, 64, 64>(c10::BFloat16*, at::native::(anonymous namespace)::CatArrInp | 155.8 | 0.6% |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<c10 | 145.5 | 0.6% |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() | 119.8 | 0.5% |
| void at::native::_scatter_gather_elementwise_kernel<128, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<2> >::operator()<at:: | 110.6 | 0.4% |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{la | 102.8 | 0.4% |
| void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::A | 77.8 | 0.3% |
| sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas | 76.9 | 0.3% |
| void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const:: | 71.2 | 0.3% |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<c10::BFloat | 68.0 | 0.3% |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>(at::TensorIteratorBase&, float) | 59.1 | 0.2% |
| void at::native::_scatter_gather_elementwise_kernel<128, 4, at::native::_cuda_scatter_gather_internal_kernel<true, at::native::OpaqueType<2> >::operator()<at::n | 57.0 | 0.2% |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_put_kernel_impl<at::native::OpaqueType<2> >(at::TensorIterator& | 55.9 | 0.2% |
| void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::silu_backward_kernel(at::TensorIteratorBase&)::{lambda()#1}::operator()() c | 40.3 | 0.2% |
| sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_off_kernel__5x_cublas | 39.9 | 0.2% |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<f | 39.6 | 0.2% |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<float>, at: | 38.5 | 0.1% |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4> >(at::native::Reduce | 33.5 | 0.1% |
| void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::A | 32.5 | 0.1% |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<2> >(at::TensorIteratorBase& | 31.2 | 0.1% |
| void cutlass::Kernel<cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1>(cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1::Params) | 29.8 | 0.1% |
| void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor | 24.0 | 0.1% |
| void cutlass::Kernel<cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_nn_align1>(cutlass_75_tensorop_bf16_s1688gemm_bf16_64x64_nn_align1::Params) | 23.1 | 0.1% |
| void (anonymous namespace)::softmax_warp_forward<c10::BFloat16, c10::BFloat16, float, 10, false, false>(c10::BFloat16*, c10::BFloat16 const*, int, int, int, boo | 18.0 | 0.1% |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at | 16.2 | 0.1% |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() | 15.8 | 0.1% |

| cpu op (inclusive) | ms/step |
|---|---|
| autograd::engine::evaluate_function: CheckpointFunctionBackward | 15457.8 |
| CheckpointFunctionBackward | 15457.0 |
| autograd::engine::evaluate_function: torch::autograd::AccumulateGrad | 10360.8 |
| aten::nonzero | 8810.7 |
| CheckpointFunction | 5585.7 |
| aten::item | 240.9 |
| aten::_local_scalar_dense | 240.5 |
| autograd::engine::evaluate_function: PreBackwardFunctionBackward | 201.3 |
| PreBackwardFunctionBackward | 197.2 |
| aten::matmul | 165.0 |
| aten::mm | 146.2 |
| LinearFunctionForZeroStage3 | 133.9 |
| autograd::engine::evaluate_function: PostBackwardFunctionBackward | 120.8 |
| PostBackwardFunctionBackward | 108.8 |
| autograd::engine::evaluate_function: LinearFunctionForZeroStage3Backward | 95.8 |
| c10d::_allgather_base_ | 95.3 |
| LinearFunctionForZeroStage3Backward | 91.8 |
| aten::mul | 56.9 |
| record_param_comms | 56.7 |
| c10d::_reduce_scatter_base_ | 55.8 |
| aten::copy_ | 46.5 |
| aten::to | 38.5 |
| PreBackwardFunction | 36.8 |
| IndexFirstAxis | 36.5 |
| autograd::engine::evaluate_function: MulBackward0 | 36.0 |
| aten::_to_copy | 35.4 |
| aten::empty | 34.0 |
| aten::slice | 31.3 |
| aten::narrow | 30.7 |
| PostBackwardFunction | 28.0 |
| aten::index | 24.5 |
| autograd::engine::evaluate_function: IndexFirstAxisBackward | 23.3 |
| aten::reshape | 22.3 |
| IndexFirstAxisBackward | 22.1 |
| MulBackward0 | 20.5 |
| aten::add | 20.2 |
| aten::sum | 19.3 |
| aten::add_ | 18.7 |
| aten::detach | 16.8 |
| aten::view | 16.7 |

