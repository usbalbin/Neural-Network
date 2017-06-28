#if 0


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
Vector<T> operator+(Vector<T>& other) {
	assert(this->size() == other.size());

	Vector<T> result(thisV.size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size(), [&](size_t i) {
		result[i] = data[i] + other[i];
	}).wait();
#else
	for (size_t i = 0; i < thisV.size(); ++i)
		result[i] = data[i] + other[i];
#endif
	return result;
}

template<typename T>
Vector<T> operator-(Vector<T>& other) {
	assert(this->size() == other.size());

	Vector<T> result(this->size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size(), [&](size_t i) {
		result[i] = data[i] - other[i];
	}).wait();
#else
	for (size_t i = 0; i < this->size(); ++i)
		result[i] = data[i] - other[i];
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



#include "VectorAvx.cpp"

#define VEC_INC
#endif