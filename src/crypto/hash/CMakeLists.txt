project(mcmhashlib)
find_package(CUDA REQUIRED)
include_directories(/usr/include/)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/cpu/)
include_directories(${CMAKE_SOURCE_DIR}/cuda/)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -gencode=arch=compute_37,code=sm_37
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_60,code=sm_60
        -gencode=arch=compute_61,code=sm_61
	-lcudart -lcudadevrt -lcuda -Xptxas -O3
        -Wno-deprecated-gpu-targets -default-stream per-thread
        --ptxas-options=-v)
#CPU
set(CPU_SRC ${CMAKE_SOURCE_DIR}/cpu/md2.c
		    ${CMAKE_SOURCE_DIR}/cpu/md5.c
		    ${CMAKE_SOURCE_DIR}/cpu/sha1.c
		    ${CMAKE_SOURCE_DIR}/cpu/sha256.c)
#Cuda
set(CU_SRC ${CMAKE_SOURCE_DIR}/cuda/md2.cu
		   ${CMAKE_SOURCE_DIR}/cuda/md5.cu
		   ${CMAKE_SOURCE_DIR}/cuda/sha1.cu
		   ${CMAKE_SOURCE_DIR}/cuda/sha256.cu)
#OCL (reserved)
#Test
set(HASHLIB_SRC ${CMAKE_SOURCE_DIR}/mcmhash.c
				${CU_SRC}
				${CPU_SRC})
set(UNIT_TEST_SRC test.c)

cuda_add_library(mcmhash SHARED ${HASHLIB_SRC})

add_executable(test_hash ${UNIT_TEST_SRC})
target_link_libraries(test_hash mcmhash)


