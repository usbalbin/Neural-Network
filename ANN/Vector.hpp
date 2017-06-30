#pragma once
#pragma once
#include <assert.h>
#include <vector>

#include <string>
#include <random>


template<typename T>
class Matrix;


#include "OCL\cl2.hpp"

template<typename T>
class Vector
{
public:
	Vector<T>() : elemCount(0) { if(!VectorOps::initialized) VectorOps::init();  }

	//TODO: Look if it can be WRITE_ONLY
	Vector<T>(size_t elemCount) : Vector<T>(){
		this->elemCount = elemCount;
		buffer = cl::Buffer(CL_MEM_READ_WRITE, sizeof(T) * elemCount);
	}


	//TODO: Look if it can be READ_ONLY
	Vector<T>(std::vector<T>& v) : Vector<T>() {//Construct from std::vector
		this->elemCount = v.size();
		this->buffer = cl::Buffer(v.begin(), v.end(), true);
	}


	Vector<T>(const std::initializer_list<T>& elems) : Vector<T>(std::vector<T>(elems)) {

	}

	Vector<T>(size_t elemCount, T value) : Vector(std::vector<T>(elemCount, value)) {}




	/*friend void swap(VectorBase<T>& first, VectorBase<T>& second) {
	std::swap(first.data, second.data);
	std::swap(first.elemCount, second.elemCount);
	std::swap(first.totalCapacity, second.totalCapacity);
	}*/

	/*VectorBase<T>(VectorBase<T>&& v) : VectorBase<T>() {	//Move constructor
	swap(*this, v);
	}*/

	/*VectorBase<T>& operator=(VectorBase<T> other) {
	swap(*this, other);
	return *this;
	}*/


	Vector<T>(size_t columnCount, std::function<double(void)>& randomizer) : Vector<T>(normalDistributed(randomizer, columnCount)) {  }

	std::vector<T> normalDistributed(std::function<double(void)>& randomizer, size_t columnCount) {
		std::vector<T> result(columnCount);
		for (auto& elem : result) {
			elem = T(randomizer());
		}
		return result;
	}


	T& operator[](size_t i) { assert(i >= 0 && i < elemCount); return data[i]; }
	const T& operator[](size_t i) const { assert(i >= 0 && i < elemCount); return data[i]; }


	constexpr size_t size() const { return elemCount; }

	constexpr size_t capacity() const { return totalCapacity; }

	std::string toString() const {
		std::stringstream ss;
		ss << "{ " << data[0];
		for (size_t i = 1; i < size(); i++)
			ss << ", " << data[i];
		ss << " }";

		return ss.str();
	}

	Vector operator+(Vector& rhs) {
		assert(rhs.size() == size());

		Vector<T> result(rhs.size());
		VectorOps::add(clRange(), result.buffer, buffer, rhs.buffer);

		return result;
	}

	Vector operator-(Vector& rhs) {
		assert(rhs.size() == size());

		Vector<T> result(rhs.size());
		VectorOps::sub(clRange(), result.buffer, buffer, rhs.buffer);

		return result;
	}

	Vector operator*(Vector& rhs) {
		assert(rhs.size() == size());

		Vector<T> result(rhs.size());
		VectorOps::mul(clRange(), result.buffer, buffer, rhs.buffer);

		return result;
	}

	friend Vector operator*(T& lhs, Vector& rhs) {
		Vector<T> result(rhs.size());
		VectorOps::mulS(result.clRange(), result.buffer, lhs, rhs.buffer);

		return result;
	}

	Vector operator/(Vector& rhs) {
		assert(rhs.size() == size());

		Vector<T> result(rhs.size());
		VectorOps::div(clRange(), result.buffer, buffer, rhs.buffer);

		return result;
	}

	Vector& operator+=(Vector& rhs) {
		assert(rhs.size() == size());

		VectorOps::addE(clRange(), buffer, rhs.buffer);

		return *this;
	}

	Vector& operator-=(Vector& rhs) {
		assert(rhs.size() == size());

		VectorOps::subE(clRange(), buffer, rhs.buffer);

		return *this;
	}

	Vector& operator*=(Vector& rhs) {
		assert(rhs.size() == size());

		VectorOps::mulE(clRange(), buffer, rhs.buffer);

		return *this;
	}

	Vector& operator/=(Vector& rhs) {
		assert(rhs.size() == size());

		VectorOps::divE(clRange(), buffer, rhs.buffer);

		return *this;
	}


	friend float sumPow2(Vector data) {
		assert(data.size() % 2 == 0, "Make sure of pow2Ness");
		using VectorOps::globalSize;
		using VectorOps::localSize;

		Vector resVec(VectorOps::workGroupCount);
		VectorOps::sumPow2(
			cl::EnqueueArgs(cl::NDRange(globalSize), cl::NDRange(localSize)),
			data.getBuffer(), resVec.getBuffer(), data.size());

		auto results = resVec.toStd();
		return std::accumulate(results.begin(), results.end(), 0.0f);
	}

	friend Vector<T> sigmoid(Vector<T>& in) {
		Vector<T> result(in.size());
		VectorOps::sigmoid(in.clRange(), result.buffer, in.buffer);
		return result;
	}

	friend Vector<T> sigmoidPrime(Vector<T>& in) {
		Vector<T> result(in.size());
		VectorOps::sigmoidPrime(in.clRange(), result.buffer, in.buffer);
		return result;
	}

	std::vector<T> toStd() { std::vector<T> result(elemCount);  cl::copy(buffer, result.begin(), result.end()); return result; }//Performs deep copy

	auto clRange() {
		return cl::EnqueueArgs(cl::NDRange(size()));
	}

	auto& getBuffer() { return buffer; }
protected:

private:
	cl::Buffer buffer;
	size_t elemCount;
	size_t totalCapacity;
};

namespace VectorOps {
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> add;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> sub;
	extern cl::KernelFunctor<cl::Buffer, float, cl::Buffer> mulS;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> mul;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> div;

	extern cl::KernelFunctor<cl::Buffer, cl::Buffer> addE;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer> subE;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer> mulE;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer> divE;

	extern cl::KernelFunctor<cl::Buffer, cl::Buffer> sigmoid;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer> sigmoidPrime;

	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> mulVM;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> mulVTM;
	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl_int> mulCR;

	extern cl::KernelFunctor<cl::Buffer, cl::Buffer, cl_int> sumPow2;

	extern size_t globalSize;
	extern size_t localSize;
	extern size_t workGroupCount;

	extern bool initialized;

	void init(cl_uint deviceType = CL_DEVICE_TYPE_GPU);
}