#include "Vector.hpp"

#include <iostream>

cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>			VectorOps::add;
cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>			VectorOps::sub;
cl::KernelFunctor<cl::Buffer, float, cl::Buffer>				VectorOps::mulS;
cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>			VectorOps::mul;
cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>			VectorOps::div;

cl::KernelFunctor<cl::Buffer, cl::Buffer>						VectorOps::addE;
cl::KernelFunctor<cl::Buffer, cl::Buffer>						VectorOps::subE;
cl::KernelFunctor<cl::Buffer, cl::Buffer>						VectorOps::mulE;
cl::KernelFunctor<cl::Buffer, cl::Buffer>						VectorOps::divE;

cl::KernelFunctor<cl::Buffer, cl::Buffer>						VectorOps::sigmoid;
cl::KernelFunctor<cl::Buffer, cl::Buffer>						VectorOps::sigmoidPrime;

cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int>	VectorOps::mulVM;
cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int>	VectorOps::mulVTM;
cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int>	VectorOps::mulCR;

//cl::KernelFunctor<cl::Buffer>									VectorOps::sum;
cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int>				VectorOps::sumPow2;

size_t VectorOps::globalSize;
size_t VectorOps::localSize;
size_t VectorOps::workGroupCount;

bool VectorOps::initialized = false;

void VectorOps::init(cl_uint deviceType) {
	cl::Context::setDefault(deviceType);

	globalSize = 2560;
	localSize = 64;
	workGroupCount = globalSize / localSize;




	std::string kernelV = "kernel void ";

	std::string kernel2Header =
		"(global float* C, global float* A){	\n\
				";

	std::string kernel3Header =
		"(global float* C, global float* A, global float* B){	\n\
				";
	std::string kernelEnd =
		"																		\n\
			}\n";
	std::string defsStr = 
	"#define i get_global_id(0)			\n"
	"#define gz get_global_size(0)		\n"
	"#define lid get_local_id(0)		\n"
	"#define lz get_local_size(0)		\n"
	"#define wgid get_group_id(0)		\n";

	std::string addStr = kernelV + "add" + kernel3Header + "C[i] = A[i] + B[i];" + kernelEnd;
	std::string subStr = kernelV + "sub" + kernel3Header + "C[i] = A[i] - B[i];" + kernelEnd;
	std::string mulStr = kernelV + "mul" + kernel3Header + "C[i] = A[i] * B[i];" + kernelEnd;
	std::string divStr = kernelV + "div" + kernel3Header + "C[i] = A[i] / B[i];" + kernelEnd;

	std::string addEStr = kernelV + "addE" + kernel2Header + "C[i] += A[i];" + kernelEnd;
	std::string subEStr = kernelV + "subE" + kernel2Header + "C[i] -= A[i];" + kernelEnd;
	std::string mulEStr = kernelV + "mulE" + kernel2Header + "C[i] *= A[i];" + kernelEnd;
	std::string divEStr = kernelV + "divE" + kernel2Header + "C[i] /= A[i];" + kernelEnd;

	std::string sigmoidStr = kernelV + "sigmoid" + kernel2Header + "C[i] = 1.0f / (1.0f + exp(-A[i]));" + kernelEnd;
	std::string sigmoidPrimeStr = kernelV + "sigmoidPrime" + kernel2Header +
		"float sig = 1.0f / (1.0f + exp(-A[i]));							\n\
		C[i] = sig * (1 - sig);"
		+ kernelEnd;

	std::string mulSStr = kernelV + "mulS(global float* C, float a, global float* B){ C[i] = a * B[i];" + kernelEnd;







	std::string mulVMStr =
	"#define column i																			\n\
	#define columnCount get_global_size(0)														\n\
		kernel void mulVM(global float* res, global float* v1, global float* m2, int rowCount) {\n\
		float temp = 0;																			\n\
		for (int row = 0; row < rowCount; ++row) {												\n\
			int mIndex = columnCount * row + column;											\n\
			temp += v1[row] * m2[mIndex];														\n\
		}																						\n\
		res[column] = temp;																		\n\
	}																							\n\
	#undef column																				\n\
	#undef columnCount\n";


	std::string mulVTMStr =
	"#define row i																				\n\
	#define rowCount get_global_size(0)															\n\
	kernel void mulVTM(global float* res, global float* v1, global float* m2, int columnCount) {\n\
		float temp = 0;																			\n\
		for (int column = 0; column < columnCount; ++column) {									\n\
			int mIndex = columnCount * row + column;											\n\
			temp += v1[column] * m2[mIndex];													\n\
		}																						\n\
		res[row] = temp;																		\n\
	}																							\n\
	#undef row																					\n\
	#undef rowCount\n";
	


	std::string mulCRStr =
	"#define row i																				\n\
	#define rowCount get_global_size(0)															\n\
	kernel void mulCR(global float* m, global float* c, global float* r, int columnCount) {		\n\
		for (int column = 0; column < columnCount; ++column) {									\n\
			int mIndex = columnCount * row + column;											\n\
			m[mIndex] = c[row] * r[column];														\n\
		}																						\n\
	}																							\n\
	#undef row																					\n\
	#undef rowCount\n";







	std::string sumPow2Str =
		"kernel void sumPow2(																	\n\
			global const float* data,															\n\
			global float* results,																\n\
			int count) {																		\n\
																								\n\
#if __OPENCL_VERSION__ < 200																	\n\
		local float temp[64];//lz];																\n\
#endif																							\n\
		float value = 0;																		\n\
		for (int globalIndex = i; globalIndex < count; globalIndex += gz) {						\n\
			value += data[globalIndex];															\n\
		}																						\n\
																								\n\
#if __OPENCL_VERSION__ >= 200																	\n\
		results[wgid] = work_group_reduce_sum(value);											\n\
#else																							\n\
		temp[lid] = value;																		\n\
		barrier(CLK_LOCAL_MEM_FENCE);															\n\
																								\n\
		for (int offset = lz / 2; offset > 0; offset /= 2) {									\n\
			if (lid < offset)																	\n\
				temp[lid] += temp[lid + offset];												\n\
			barrier(CLK_LOCAL_MEM_FENCE);														\n\
		}																						\n\
																								\n\
		if (lid == 0)																			\n\
			results[wgid] = temp[0];															\n\
																								\n\
#endif																							\n\
	}\n";

	std::vector<std::string> sources{ defsStr,
		addStr, subStr, mulStr, divStr,
		addEStr, subEStr, mulEStr, divEStr,
		sigmoidStr, sigmoidPrimeStr,
		mulSStr,
		mulVMStr, mulVTMStr, mulCRStr,
		sumPow2Str
	};
	
	cl_int status = CL_SUCCESS;
	cl::Program program(sources, &status);
	
	try {
		program.build("");// "-cl-std=CL2.0");
	}
	catch (...) {
		auto deviceStatuses = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
		for (auto& deviceStatus : deviceStatuses) {
			std::cout << "Device: " << deviceStatus.first.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
			std::cout << deviceStatus.second << std::endl << std::endl;
		}
		throw "Sten";
	}

	auto deviceStatuses = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
	for (auto& deviceStatus : deviceStatuses) {
		std::cout << "Device: " << deviceStatus.first.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
		std::cout << deviceStatus.second << std::endl << std::endl;
	}

	std::cout << "Done compiling" << std::endl;
	std::cout << "Creaing Kernels..." << std::endl;

	add = decltype(add)(program, "add");
	sub = decltype(sub)(program, "sub");
	mul = decltype(mul)(program, "mul");
	div = decltype(div)(program, "div");

	addE = decltype(addE)(program, "addE");
	subE = decltype(subE)(program, "subE");
	mulE = decltype(mulE)(program, "mulE");
	divE = decltype(divE)(program, "divE");

	sigmoid = decltype(sigmoid)(program, "sigmoid");
	sigmoidPrime = decltype(sigmoidPrime)(program, "sigmoidPrime");

	mulS = decltype(mulS)(program, "mulS");

	mulVM = decltype(mulVM)(program, "mulVM");
	mulVTM = decltype(mulVTM)(program, "mulVTM");
	mulCR = decltype(mulCR)(program, "mulCR");

	sumPow2 = decltype(sumPow2)(program, "sumPow2");

	std::cout << "OpenCL initialized" << std::endl;

	initialized = true;
}