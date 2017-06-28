#include "Utils.hpp"

#include <ammintrin.h>
#include <immintrin.h>

//https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-float-vector-sum-on-x86/35270026#35270026
float hsum_ps_sse3(const __m128& v) {
	__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

//https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-float-vector-sum-on-x86/35270026#35270026
float hsum256_ps_avx(const __m256& v) {
	__m128 vlow = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
	vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
	return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
									   // (no wasted instructions, and all of them are the 4B minimum)
}


bool isPow2(unsigned long long x) {
	return x && !(x & (x - 1));
}