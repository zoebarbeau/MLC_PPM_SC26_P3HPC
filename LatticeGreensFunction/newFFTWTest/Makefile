# FFTW_HOME=/home/sanilr/fftw-3.3.10
CC=clang++
# NVCC=nvcc
# HIPCC=hipcc
# INCLUDE=${FFTX_HOME}/include
# LIB=${FFTX_HOME}/lib
# export LD_LIBRARY_PATH=/home/h82/Documents/Bluestone/SPIRAL/FFTX/fftx/lib:$LD_LIBRARY_PATH
DIM=3
USE_HDF5=TRUE
PROTO=/g/g16/barbeau2/CPU/proto
CXXOPTFLAGS+= -DPR_HDF5=TRUE
HDF5_LIB=/usr/local/lib
HDF5_INC=/usr/local/include
CPPFLAGS:=-DDIM=$(DIM) -I$(PROTO)/include -I$(PROTO)/include/base -I$(PROTO)/include/ops -I$(PROTO)/include/utils -I/$(HDF5_INC) -I$(PROTO)/include/ops -I/ -I. -I../src -I../../_common     
all: test

test:
	$(CC) -DDIM=$(DIM) -DPR_HDF5 -I$(PROTO)/include -I$(PROTO)/include/base -I$(HDF5_INC) -I../src -std=c++17 -O3 -w FFTW_newtest.cpp -o cpu_fftw_test -lfftw3 -L$(HDF5_LIB) -lhdf5 -lz
# test:
# 	$(CC) -std=c++17 -O3 -w -I$(INCLUDE) -DFFTX_CPU FFTX_rconv_test.cpp -o cpu_fftx_test -L$(LIB) -lfftx_dftbat_cpu -lfftx_idftbat_cpu -lfftx_imddft_cpu -lfftx_imdprdft_cpu -lfftx_iprdftbat_cpu -lfftx_mddft_cpu -lfftx_mdprdft_cpu -lfftx_prdftbat_cpu -lfftx_rconv_cpu

# cuda:
# 	$(NVCC) -O3 -w -I$(INCLUDE) -DFFTX_CUDA hockney_test_cuda_timer.cu -o cuda_hockney -L$(LIB) -lcuda -lnvrtc -lcufft -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_rconv_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu

# hip:
# 	$(HIPCC) -O3 -w -I$(INCLUDE) -DFFTX_HIP hockney_test_hip.cpp -o hip_hockney -L$(LIB) -lrocfft -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_rconv_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu 

clean:
	rm *_fftw_test
