#pragma once

#include "Vector.hpp"
#include <vector>
#include <string>
#include <sstream>

template<typename T>
using Row = Vector<T>;

template<typename T>
class Column;

#if 0

template<typename T>
class Matrix
{
public:
	Matrix() {};

	Matrix(size_t rowCount, size_t columnCount) : rows(rowCount, Row<T>(columnCount)) { }
	
	Matrix(size_t rowCount, size_t columnCount, std::function<double(void)>& randomizer) :
		rows(rowCount, Row<T>(columnCount)) {
		for (auto& r : rows) 
			r.normalDistribute(randomizer);
	}

	Matrix(size_t rowCount, size_t columnCount, T value) : rows(rowCount, Row<T>(columnCount, value )) { }


	template<typename T>
	Matrix(const std::vector<std::vector<T>>& data) { for (auto& row : data) rows.push_back(row); }

	~Matrix() = default;



	inline Column<T> col(size_t i) {
		return Column<T>(*this, i);
	}


	inline Row<T>& row(size_t i) { return rows[i]; }


	inline Row<T>& operator[](size_t i) { return row(i); }

	Matrix<T> operator*(Matrix<T>& other) { return mul(*this, other); }

	Matrix<T>& operator+=(Matrix<T>& other) {
		assert(
			this->getRowCount() == other.getRowCount() &&
			this->getColumnCount() == other.getColumnCount()
		);


		for (size_t row = 0; row < getRowCount(); row++)
			rows[row] += other[row];
		return *this;
	}

	Matrix<T> operator-=(Matrix<T>& other) {
		assert(
			this->getRowCount() == other.getRowCount() &&
			this->getColumnCount() == other.getColumnCount()
		);
		
		//Don't multithread further

		Matrix<T> result(this->getRowCount(), this->getColumnCount());
		for (size_t row = 0; row < result.getRowCount(); row++)
			rows[row] -= other[row];

		return *this;
	}

	std::string toString() const {
		std::stringstream ss;
		ss << "{ " << rows[0].toString() << " ";
		for (size_t i = 1; i < rows.size(); i++)
			ss << std::endl << "  " << rows[i].toString() << " ";
		ss << "}";

		return ss.str();
	}

	constexpr size_t getColumnCount() const { return rows[0].size(); }
	constexpr size_t getRowCount() const { return rows.size(); }
private:
	std::vector<Row<T>> rows;			//Change this to a container that wont copy the first element on construction
};

template<typename T>
Matrix<T> mul(
	Matrix<T>& thisM,
	Matrix<T>& otherM
) {
	assert(thisM.getColumnCount() == otherM.getRowCount());
	
	Matrix<T> result(thisM.getRowCount(), otherM.getColumnCount());
	
#ifdef MULTI_THREADED
	ThreadThing t;
	t.run(result.getRowCount(), result.getColumnCount(), [&](size_t row, size_t col) {
		result[row][col] = dot(thisM.row(row), otherM.col(col));
	});
	t.wait();
#else
	for (size_t row = 0; row < result.getRowCount(); row++)
		for (size_t col = 0; col < result.getColumnCount(); col++)
			result[row][col] = dot(thisM.row(row), otherM.col(col));
#endif
	return result;
}

template<typename T>
Matrix<T> operator*(
	T& thisS,
	Matrix<T>& otherM
) {
	Matrix<T> result(otherM.getRowCount(), otherM.getColumnCount());
	for (size_t row = 0; row < result.getRowCount(); row++)
		result[row] = thisS * otherM.row(row);
	return result;
}

template<typename T>
Matrix<T> add(
	Matrix<T>& thisM,
	Matrix<T>& otherM
) {
	assert(
		thisM.getRowCount() == otherM.getRowCount() &&
		thisM.getColumnCount() == otherM.getColumnCount()
	);
	

	Matrix<T> result(thisM.getRowCount(), thisM.getColumnCount());
	for (size_t row = 0; row < result.getRowCount(); row++)
		result[row] = thisM[row] + otherM[row];
	return result;
}

template<typename T>
class Column {
public:
	template<typename T>
	inline Column(Matrix<T>& m, size_t columnIndex) : data(m), columnIndex(columnIndex) {};

	//template<typename T, size_t rowCount, size_t columnCount>
	T& operator[](size_t i) {
		return data[i][columnIndex];
	}

	size_t size() { return data.getRowCount(); }
private:
	Matrix<T>& data;
	size_t columnIndex;
};

//TODO: look into wether this can be vectorized
template<typename T>
T dot(Vector<T>& thisV, Column<T>& otherC) {
	assert(thisV.size() == otherC.size());

	T result = T(0);
	for (size_t i = 0; i < thisV.size(); i++)
		result += thisV[i] * otherC[i];

	return result;
}
#else

template<typename T>
class Matrix
{
public:
	Matrix() {};

	Matrix(size_t rowCount, size_t columnCount) : rowCount(rowCount), columnCount(columnCount), data(rowCount * columnCount) { }

	Matrix(size_t rowCount, size_t columnCount, std::function<double(void)>& randomizer) :
		rowCount(rowCount), columnCount(columnCount),
		data(rowCount * columnCount, randomizer) {
	}

	Matrix(size_t rowCount, size_t columnCount, std::vector<T>& values) : rowCount(rowCount), columnCount(columnCount), data(values) { }

	Matrix(size_t rowCount, size_t columnCount, T value) : rowCount(rowCount), columnCount(columnCount), data(rowCount * columnCount, value) { }

	~Matrix() = default;



	//Matrix<T> operator*(Matrix<T>& other) { Matrix VectorOps::mulMM(*this, other); }


	friend Matrix<T> operator*(T& lhs, Matrix<T>& rhs) {
		Matrix<T> result(rhs.getRowCount(), rhs.getColumnCount());
		result.data = lhs * rhs.data;		//mul elementwise
		return result;
	}

	Matrix<T>& operator+=(Matrix<T>& other) {
		assert(
			this->getRowCount() == other.getRowCount() &&
			this->getColumnCount() == other.getColumnCount()
		);

		this->data += other.data;		//Add elementwise

		return *this;
	}

	Matrix<T> operator-=(Matrix<T>& other) {
		assert(
			this->getRowCount() == other.getRowCount() &&
			this->getColumnCount() == other.getColumnCount()
		);

		this->data -= other.data;		//Sub elementwise

		return *this;
	}

	std::string toString() const {
		std::stringstream ss;
		ss << "{ " << rows[0].toString() << " ";
		for (size_t i = 1; i < rows.size(); i++)
			ss << std::endl << "  " << rows[i].toString() << " ";
		ss << "}";

		return ss.str();
	}

	auto& getBuffer() { return data.getBuffer(); }
	constexpr size_t getColumnCount() const { return columnCount; }
	constexpr size_t getRowCount() const { return rowCount; }
	Vector<T>& getData() { return data; }//TODO: Consider removing me

	inline void writeToFile(std::ostream& file) {
		writeIntToFile(rowCount, file);
		writeIntToFile(columnCount, file);

		auto v = data.toStd();
		for (auto elem : v)
			writeFloatToFile(elem, file);
	}

	Matrix(std::istream& file) {
		this->rowCount = readIntFromFile(file);
		this->columnCount = readIntFromFile(file);

		std::vector<T> v;
		for (int i = 0; i < rowCount; ++i)
			for (int j = 0; j < columnCount; ++j)
				v.push_back(readFloatFromFile(file));

		this->data = Vector<T>(v);
	}
private:
	size_t rowCount;
	size_t columnCount;
	Vector<T> data;			//Change this to a container that wont copy the first element on construction
};

#endif