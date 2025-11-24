nvcc -O2 -std=c++14 -lcurand -arch=native gpu_dwconv2d_inference.cu cpu_dwconv2d_inference_ref.cu test_dwconv2d.cu -o test_dwconv2d
./test_dwconv2d

### Only use this if you have permission to open the program counter for profiling.
# nsys profile -w true -t cuda,nvtx,osrt --force-overwrite=true --stats=true --gpu-metrics-device=0 -x true -o test_dwconv2d_nsys_profile ./test_dwconv2d
# ncu -f -o test_dwconv2d_ncu_profile --kernel-name gpu_dwconv2d_nchw --launch-count 1 --set full --cache-control none --import-source true --target-processes all ./test_dwconv2d
