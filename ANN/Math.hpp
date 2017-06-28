#pragma once









#include "Matrix.hpp"
#include "Vector.hpp"



#if 0


template<typename T>
Vector<T> operator*(
	Vector<T>& thisV,
	Matrix<T>& otherM
) {
	assert(thisV.size() == otherM.getRowCount());


	Vector<T> result(otherM.getColumnCount());
#ifdef MULTI_THREADED
	ThreadPool::loop(result.size(), [&](size_t col) {
		result[col] = dot(thisV, otherM.col(col));
	});
#else
	for (size_t col = 0; col < result.size(); col++)
		result[col] = dot(thisV, otherM.col(col));
#endif
	return result;
}

template<typename T>
Vector<T> mulTranspMatrix(
	Vector<T>& thisV,
	Matrix<T>& otherM
) {
	assert(thisV.size() == otherM.getColumnCount());
	//TODO: check if col and row should be swapped

	Vector<T> result(otherM.getRowCount());
#ifdef MULTI_THREADED
	ThreadPool::loop(result.size(), [&](size_t row) {
		result[row] = dot(thisV, otherM.row(row));
	});
#else
	for (size_t row = 0; row < result.size(); row++)
		result[row] = dot(thisV, otherM.row(row));
#endif
	return result;
}

template<typename T>
Vector<T> mul(
	Matrix<T>& thisM,
	Vector<T>& otherV
) {
	//TODO implement me!!
	assert(false);
}


template<typename T>
Matrix<T> mulColumnRow(Vector<T>& column, Vector<T>& row) {
	Matrix<T> result(column.size(), row.size());

#ifdef MULTI_THREADED
	ThreadPool::loop(result.getRowCount(), [&](size_t rowIndex) {
		for(size_t colIndex = 0; colIndex < result.getColumnCount(); ++colIndex)
			result[rowIndex][colIndex] = column[rowIndex] * row[colIndex];
	});
#else
	for (size_t rowIndex = 0; rowIndex < result.getRowCount(); rowIndex++)
		for (size_t colIndex = 0; colIndex < result.getColumnCount(); colIndex++)
			result[rowIndex][colIndex] = column[rowIndex] * row[colIndex];
#endif
	return result;
}


#else

template<typename T>
Vector<T> operator*(
	Vector<T>& thisV,
	Matrix<T>& otherM
	) {
	assert(thisV.size() == otherM.getRowCount());

	Vector<T> result(otherM.getColumnCount());
	VectorOps::mulVM(result.clRange(), result.getBuffer(), thisV.getBuffer(), otherM.getBuffer(), otherM.getRowCount());

	return result;
}


template<typename T>
Vector<T> mulTranspMatrix(
	Vector<T>& thisV,
	Matrix<T>& otherM
) {
	assert(thisV.size() == otherM.getColumnCount());
	//TODO: check if col and row should be swapped

	Vector<T> result(otherM.getRowCount());

	VectorOps::mulVTM(result.clRange(), result.getBuffer(), thisV.getBuffer(), otherM.getBuffer(), otherM.getColumnCount());

	return result;
}


template<typename T>
Matrix<T> mulColumnRow(Vector<T>& column, Vector<T>& row) {
	Matrix<T> result(column.size(), row.size());
	
	VectorOps::mulCR(cl::EnqueueArgs(cl::NDRange(result.getRowCount())), result.getBuffer(), column.getBuffer(), row.getBuffer(), result.getColumnCount());

	return result;
}

#endif
//////////////


#if 0

template<>
Vector<float> mul(
	Vector<float>& thisV,
	Matrix<float>& otherM
) {
	assert(thisV.size() == otherM.getRowCount());


	Vector<float> result(otherM.getColumnCount());
#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size(), [&](size_t col) {
		result[col] = dot(thisV, otherM.col(col));
	}).wait();
#else
	for (size_t col = 0; col < result.size(); col++)
		result[col] = dot(thisV, otherM.col(col));
#endif
	return result;
}

template<>
Vector<float> mulTranspMatrix(
	Vector<float>& thisV,
	Matrix<float>& otherM
) {
	assert(thisV.size() == otherM.getColumnCount());
	//TODO: check if col and row should be swapped

	Vector<float> result(otherM.getRowCount());
#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size(), [&](size_t row) {
		result[row] = dot(thisV, otherM.row(row));
	}).wait();
#else
	for (size_t row = 0; row < result.size(); row++)
		result[row] = dot(thisV, otherM.row(row));
#endif
	return result;
}

template<>
Vector<float> mul(
	Matrix<float>& thisM,
	Vector<float>& otherV
) {
	//TODO implement me!!
	assert(false);
	return Vector<float>();
}

template<>
Vector<float> operator*(
	float& thisS,
	Vector<float>& otherV
	) {
	Vector<float> result(otherV.size());

	AvxFloat8 temp(thisS);

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.size() / 8, [&](size_t i) {
		result.getVElem(i) = temp * otherV.getVElem(i);
	}).wait();
#else
	for (size_t i = 0; i < result.size() / 8; i++)
		result.getVElem(i) = temp * otherV.getVElem(i);
#endif
	return result;
}

template<>
Matrix<float> mulColumnRow(Vector<float>& column, Vector<float>& row) {
	Matrix<float> result(column.size(), row.size());

#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.getRowCount(), result.getRowCount(), [&](size_t rowIndex, size_t colIndex) {
		result[rowIndex][colIndex] = column[rowIndex] * row[colIndex];
	}).wait();
#else
	for (size_t rowIndex = 0; rowIndex < result.getRowCount(); rowIndex++)
		for (size_t colIndex = 0; colIndex < result.getColumnCount(); colIndex++)
			result[rowIndex][colIndex] = column[rowIndex] * row[colIndex];
#endif
	return result;

}

#endif