#pragma once
#include "Vector.hpp"
#include <cmath>

/*template<typename T>
constexpr T sigmoid(T x) {
	return T(1) / (T(1) + std::exp(-x));
}

template<typename T>
constexpr Vector<T> sigmoid(Vector<T> x) {
	for(auto& elem : x)
		elem = sigmoid(elem);
	return x;
}

template<typename T>
constexpr T sigmoidPrime(T x) {
	T sig = sigmoid(x);
	return sig * (T(1) - sig);
}

template<typename T>
constexpr Vector<T> sigmoidPrime(Vector<T> x) {
	for (auto& elem : x)
		elem = sigmoidPrime(elem);
	return x;
}*/

float hsum_ps_sse3(const __m128& v);
float hsum256_ps_avx(const __m256& v);

inline void writeIntToFile(int32_t value, std::ostream& file) {
	file.write((char*)&value, sizeof(int32_t));
}

inline void writeFloatToFile(float value, std::ostream& file) {
	file.write((char*)&value, sizeof(float));
}

inline int32_t readIntFromFile(std::istream& file) {
	int32_t value;
	file.read((char*)&value, sizeof(int32_t));
	return value;
}

inline float readFloatFromFile(std::istream& file) {
	float value;
	file.read((char*)&value, sizeof(float));
	return value;
}