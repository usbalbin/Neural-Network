#pragma once
#include <assert.h>
#include <vector>

#include <string>
#include <random>

#include "ThreadPool.hpp"

template<typename T>
class Matrix;

#include "VectorBase.hpp"
#if 0
template<typename T>
class Vector : public Vector<T>
{
public:
	using Vector::Vector;//Use inherited constructors

	Vector<T> operator*(Vector<T>& other) {
		return forEachRet<T>(*this, other, [](const T& lhs, const T& rhs) -> T{
			return lhs * rhs;
		});
	};



	Vector<T> operator+(Vector<T>& other) {
		return forEachRet(*this, other, [](const T& lhs, const T& rhs) {
			return lhs + rhs;
		});
	};

	Vector<T> operator-(Vector<T>& other) {
		return forEachRet(*this, other, [](const T& lhs, const T& rhs) {
			return lhs - rhs;
		});
	};



	Vector<T>& operator+=(Vector<T>& other) {
		forEach(*this, other, [](T& lhs, const T& rhs) -> void {
			lhs += rhs;
		});
		return *this;
	};

	Vector<T>& operator-=(Vector<T>& other) {
		forEach<T>(*this, other, [](T& lhs, const T& rhs) -> void{
			lhs -= rhs;
		});
		return *this;
	};

private:
};







template<typename T, typename F>
Vector<T> forEachRet(Vector<T>& lhs, Vector<T>& rhs, F& p) {
	assert(lhs.size() == rhs.size());
	Vector<T> result(lhs.size());
	forEachH([&result, &p](size_t i, Vector<T>& lhs, Vector<T>& rhs) -> void{
		result[i] = p(lhs[i], rhs[i]);
	}, lhs, rhs);
	return result;
}

template<typename T, typename F>
void forEach(Vector<T>& lhs, Vector<T>& rhs, F& p) {
	assert(lhs.size() == rhs.size());
	forEachH([&p](size_t i, Vector<T>& lhs, Vector<T>& rhs) {
		p(lhs[i], rhs[i]);
	}, lhs, rhs);
}


template<typename F, typename T, typename... Ts>//, typename... Ts>
void forEachH(F& p, Vector<T>& v, Ts&... arg) {
//	assert(false, "USE AVX VERSION INSTEAD!!!");
//	throw;
#ifdef MULTI_THREADED
	ThreadPool::loop(v.size(), [&](size_t i) {
		p(i, v, arg...);
	});
#else
	for (size_t i = 0; i < v.size(); i++)
		p(i, v, arg...);
#endif
}



template<typename T>
Vector<T> operator*(T& lhs, Vector<T>& rhs) {
	Vector<T> result(rhs.size());
	//forEachH([&](size_t i) {
	//	result[i] = lhs * rhs[i];
	//}, rhs, lhs);
	return result;
}

template<typename T>
T dot(Vector<T>& lhs, Vector<T>& rhs) {
	assert(lhs.size() == rhs.size());
	T result = 0;
	for (size_t i = 0; i < rhs.size(); ++i) {
		result += lhs[i] * rhs[i];
	}
	return result;
}

//#include "VectorAvx.hpp"
#include "Vector.cpp"



#endif












#if 0

#include "Vector.hpp"

template<typename T>
Vector<T> Vector<T>::elemWiseMul(Vector<T>& other) {
	assert(this->size() == other.size());

	Vector<T> result(size());
#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(size(), [&](size_t i) {
		result[i] = data[i] * other[i];
	}).wait();
#else
	for (size_t i = 0; i < size(); ++i)
		result[i] = data[i] * other[i];
#endif
	return result;
}

template<typename T>
Vector<T>& Vector<T>::operator+=(Vector<T>& other) {
	assert(this->size() == other.size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(size(), [&](size_t i) {
		data[i] += other[i];
	}).wait();
#else
	for (size_t i = 0; i < size(); i++)
		data[i] += other[i];
#endif
	return *this;
}


template<typename T>
Vector<T>& Vector<T>::operator-=(Vector<T>& other) {
	assert(this->size() == other.size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(size(), [&](size_t i) {
		data[i] -= other[i];
	}).wait();
#else
	for (size_t i = 0; i < size(); i++)
		data[i] -= other[i];
#endif
	return *this;
}

template<typename T>
Vector<T> operator*(
	T& thisS,
	Vector<T>& otherV
	) {
	Vector<T> result(otherV.size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size(), [&](size_t i) {
		result[i] = thisS * otherV[i];
	}).wait();
#else
	for (size_t i = 0; i < result.size(); i++)
		result[i] = thisS * otherV[i];
#endif
	return result;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
T dot(Vector<T>& thisV, Vector<T>& otherV) {
	assert(thisV.size() == otherV.size());

	T result = T(0);
	for (size_t i = 0; i < thisV.size(); i++)
		result += thisV[i] * otherV[i];

	return result;
}

template<typename T>
Vector<T> add(Vector<T>& thisV, Vector<T>& otherV) {
	assert(thisV.size() == otherV.size());

	Vector<T> result(thisV.size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size(), [&](size_t i) {
		result[i] = thisV[i] - otherV[i];
	}).wait();
#else
	for (size_t i = 0; i < thisV.size(); ++i)
		result[i] = thisV[i] + otherV[i];
#endif
	return result;
}

template<typename T>
Vector<T> sub(Vector<T>& thisV, Vector<T>& otherV) {
	assert(thisV.size() == otherV.size());

	Vector<T> result(thisV.size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size(), [&](size_t i) {
		result[i] = thisV[i] - otherV[i];
	}).wait();
#else
	for (size_t i = 0; i < thisV.size(); ++i)
		result[i] = thisV[i] - otherV[i];
#endif

	return result;
}


#endif