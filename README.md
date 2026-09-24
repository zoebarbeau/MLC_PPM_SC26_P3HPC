# MLC_PPM
The Method of Location Correction requires Kokkos, Cabana, and FFTX libraries. Older versions of Kokkos are most directly compatible with the other libraries. We recommend using 4.4.1. For CUDA builds on Perlmutter, we recommend using cudatoolkit/12.9 with C++ 17.

To build the repository, the following steps must be done.  
To install FFTX, you must first install  spiral. The config script and cmake of FFTX require the platform it is to be built on (HIP,CUDA, or CPU).
```
git clone https://github.com/spiral-software/fftx
cd fftx
export FFTX_HOME=`pwd`
unset SPIRAL_HOME
source ./get_spiral.sh
./config-fftx-libs.sh <Cuda/HIP/CPU>
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -D_codegen=<CUDA/HIP/CPU> ..
make install
cd ~/spiral-software
export SPIRAL_HOME=`pwd`
```


To build Kokkos, clone the repository and checkout the 4.4.1 version. The legacy view option is needed for compatibility with Cabana. You can specify your architecture or remove that command to have the architecture detected instead. 
```
    git clone https://github.com/kokkos/kokkos.git
    git checkout 15dc143 
    export KOKKOS_SRC_DIR=`pwd`/kokkos
    export KOKKOS_INSTALL_DIR=$KOKKOS_SRC_DIR/build/install

    cd ./kokkos
    mkdir build
    cd build
```
For CUDA use:
```
    cmake \
      -D CMAKE_BUILD_TYPE="Release" \
      -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
      -D CMAKE_CXX_COMPILER="${KOKKOS_SRC_DIR}/bin/nvcc_wrapper" \
      -D CUDAToolkit_ROOT="${CUDA12_ROOT}" \
      -D Kokkos_ENABLE_SERIAL=ON \
      -D Kokkos_ENABLE_OPENMP=ON \
      -D Kokkos_ENABLE_CUDA=ON \
      -D Kokkos_ENABLE_CUDA_LAMBDA=ON \
      -D Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON \
      -D Kokkos_ARCH_<Your Architecture>=ON \
      \
      .. ;
    make install
    cd ../.. 
```
For HIP use:
```
    cmake \
      -D CMAKE_BUILD_TYPE="Release" \
      -D CMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL_DIR \
      -D CMAKE_CXX_COMPILER=hipcc \
      -D Kokkos_ENABLE_SERIAL=ON \
      -D Kokkos_ENABLE_OPENMP=ON \
      -D Kokkos_ENABLE_HIP=ON \
      -D Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON \
      -D Kokkos_ARCH_<Your Architecture>=ON \
      \
      .. ;
    make install
    cd ../.. 
```

To build Cabana, make sure to turn testing off. Requiring HIP or CUDA checks if the Kokkos build is appropriate. 

```
git clone https://github.com/ECP-copa/Cabana.git
export KOKKOS_SRC_DIR=`pwd`/kokkos
export KOKKOS_INSTALL_DIR=`pwd`/kokkos/build/install
export CABANA_INSTALL_DIR=<=`pwd`/Cabana/build/install
cd ~/Cabana
mkdir build; cd build
cmake \
-D CMAKE_BUILD_TYPE="Debug" \
-D CMAKE_PREFIX_PATH=$KOKKOS_INSTALL_DIR \
-D CMAKE_INSTALL_PREFIX=$CABANA_INSTALL_DIR \
-DCMAKE_CXX_COMPILER="${KOKKOS_SRC_DIR}/bin/nvcc_wrapper" \
-DKokkos_ROOT="${KOKKOS_INSTALL_DIR}" \
-D Cabana_REQUIRE_<CUDA/HIP>=ON \
-D Cabana_REQUIRE_MPI=ON \
-DMPI_CXX_COMPILER="$(which mpicxx)" \
-DCabana_INSTALL_PACKAGEDIR=`pwd` \
-DCabana_ENABLE_TESTING=OFF \
..;
Make install

To build the MLC repo on for CUDA, specify the nvcc wrapper for CXX. Similarly, specify hipcc for HIP.
```
git clone https://github.com/zoebarbeau/MLC_PPM_SC26_P3HPC.git
cd MLC_PPM_SC26_P3HPC
git checkout Cleaned_GPU

cmake \
-D CMAKE_BUILD_TYPE="Debug" \
-DCMAKE_CXX_COMPILER="${KOKKOS_SRC_DIR}/bin/nvcc_wrapper" \
-DCMAKE_PREFIX_PATH="${CABANA_INSTALL_DIR};${FFTX_HOME}" \
-D CMAKE_INSTALL_PREFIX=install \
-DMPI_CXX_COMPILER="$(which mpicxx)" \
..;
make Hill

```
The test case given is the Hill's vortex which is a spherical vortex with an analytical solution. The code outputs the L2 particle velocity error, L2 velocity on the grid error, and maximum velocity error as well as the average time over 15 calls for the four performance kernels of Depositions, Convolutions, Corrections, and Interactions. The code can be run as:
```
./Hill <grid spacing> <cuda/hip/serial/openmp> <particle spacing> <optimization> <correction radius>
```
Correction radius is an optional argument that defaults to 4. The grid spacing is the grid discretization and the particle spacing is the initial spacing of the particles. To have a two particles per direction, run with particle spacing that is half of the grid spacing and so forth. The two options for optimization are: 1. base, 2. optimization. These correspond the base and optimized cases in the paper.  An example run is:

```
./Hill 0.03125 cuda 0.015625 base 4
```
To check correctness, the errors for this case are: 
```
Max error component: 0.0960606
L2 GRID = 0.00562218195142
L2 P = 0.00438079757796
```

