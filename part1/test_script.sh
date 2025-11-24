nvcc -O2 -std=c++14 -lcurand -arch=native test2.cu cpu_bn_inference_ref.cu test_bn.cu -o test_bn
./test_bn

### Only use this if you have permission to open the program counter for profiling.
# nsys profile -w true -t cuda,nvtx,osrt --force-overwrite=true --stats=true --gpu-metrics-device=0 -x true -o test_bn_nsys_profile ./test_bn
# ncu -f -o test_bn_ncu_profile --kernel-name gpu_bn_inference_nchw --launch-count 1 --set full --cache-control none --import-source true --target-processes all ./test_bn
