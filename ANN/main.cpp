#include "ANN.hpp"
#include <iostream>
#include "Math.hpp"
#include "ThreadPool.hpp"

#include <functional>
#include <numeric>


std::vector<float> sigmoid(std::vector<float> v) {
	for (auto& e : v)
		e = 1.0f / (1.0f + std::exp(-e));
	return v;
}

std::vector<float> sigmoidPrime(std::vector<float> v) {
	v = sigmoid(v);
	for (auto& e : v)
		e = e * (1.0f - e);
	return v;
}

std::vector<float> mul(std::vector<float> v1, std::vector<std::vector<float>> m2) {
	std::vector<float> result(m2[0].size());		//colcount
	for (size_t col = 0; col < m2[0].size(); col++) {
		float temp = 0;
		for (size_t row = 0; row < m2.size(); row++)//rowCount
			temp += v1[row] * m2[row][col];
		result[col] = temp;
	}
	return result;
}



std::vector<float> mulTranspMatrix(std::vector<float> v1, std::vector<std::vector<float>> m2) {
	std::vector<float> result(m2.size());		//rowcount
	for (size_t row = 0; row < m2.size(); row++) {
		float temp = 0;
		for (size_t col = 0; col < m2[0].size(); col++)//colCount
			temp += v1[col] * m2[row][col];
		result[row] = temp;
	}
	return result;
}


std::vector<std::vector<float>> mulColumnRow(std::vector<float>& column, std::vector<float>& row) {
	std::vector<std::vector<float>> result(column.size(), std::vector<float>(row.size()));

	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		for (size_t colIndex = 0; colIndex < result[0].size(); colIndex++)
			result[rowIndex][colIndex] = column[rowIndex] * row[colIndex];
	return result;
}

std::vector<std::vector<float>> sub(std::vector<std::vector<float>> result, std::vector<std::vector<float>>& lhs) {
	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		for (size_t colIndex = 0; colIndex < result[0].size(); colIndex++)
			result[rowIndex][colIndex] -= lhs[rowIndex][colIndex];
	return result;
}

std::vector<std::vector<float>> add(std::vector<std::vector<float>> result, std::vector<std::vector<float>>& lhs) {
	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		for (size_t colIndex = 0; colIndex < result[0].size(); colIndex++)
			result[rowIndex][colIndex] += lhs[rowIndex][colIndex];
	return result;
}

std::vector<std::vector<float>> mulS(std::vector<std::vector<float>> result, float lhs) {
	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		for (size_t colIndex = 0; colIndex < result[0].size(); colIndex++)
			result[rowIndex][colIndex] *= lhs;
	return result;
}

std::vector<float> add(std::vector<float> result, std::vector<float>& lhs) {
	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		result[rowIndex] += lhs[rowIndex];
	return result;
}
std::vector<float> sub(std::vector<float> result, std::vector<float>& lhs) {
	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		result[rowIndex] -= lhs[rowIndex];
	return result;
}
std::vector<float> mul(std::vector<float> result, std::vector<float>& lhs) {
	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		result[rowIndex] *= lhs[rowIndex];
	return result;
}
std::vector<float> mulS(std::vector<float> result, float lhs) {
	for (size_t rowIndex = 0; rowIndex < result.size(); rowIndex++)
		result[rowIndex] *= lhs;
	return result;
}

auto to2D(std::vector<float>& vec1d, int row, int col) {
	assert(vec1d.size() == row * col);
	std::vector<std::vector<float>> result;
	int index = 0;
	for (int i = 0; i < row; ++i) {
		std::vector<float> e;
		for (int j = 0; j < col; ++j)
			e.push_back(vec1d[index++]);
		result.push_back(e);
	}
	return result;
}


int main()
{
	std::mt19937 randomEngine = std::mt19937(0/*randomDevice()*/);
	std::normal_distribution<double> distribution(0, 0.25);
	std::function<double(void)> randomizer = std::bind(distribution, randomEngine);


	VectorOps::init(CL_DEVICE_TYPE_GPU);
	//ThreadPool::init(2);


	Vector<float> a(1000, randomizer);
	Matrix<float> b(1000, 500, randomizer);
	Vector<float> c(2000, randomizer);
	Matrix<float> d(500, 1000, randomizer);

	auto as = a.toStd();
	auto bs = to2D(b.getData().toStd(), 1000, 500);
	auto cs = c.toStd();
	auto ds = to2D(d.getData().toStd(), 500, 1000);


	auto op = [](float a, float b) { return a - b; };

	auto forAll = [](std::vector<float>& a, std::vector<float>& b, std::function<float(float, float)> p) {
		std::vector<float> c(a.size());
		for (size_t i = 0; i < a.size(); ++i)
			c[i] = a[i] * b[i];
		return c;
	};

	auto equal = [](std::vector<float>& a, std::vector<float>& b) {


		if (a.size() != b.size()) throw "sten";

		for (size_t i = 0; i < a.size(); ++i)
			if (a[i] != b[i]) {
				std::cout << a[i] << ", " << b[i] << std::endl;
				std::system("pause");
			}
	};

	auto equalM = [](std::vector<float>& a, std::vector<std::vector<float>>& b) {

		std::vector<float> c;
		for (auto& row : b)
			c.insert(c.end(), row.begin(), row.end());

		if (a.size() != c.size()) throw "sten";

		for (size_t i = 0; i < a.size(); ++i)
			if (a[i] != c[i])
				throw "sten";
	};

	/*std::vector<float> as{1, 2, 3, 4};
	std::vector<std::vector<float>> bs{
		{ 1, 2, 3, 4, },
		{ 5, 6, 7, 8, },
		{ 9, 1, 2, 3, },
		{ 4, 5, 6, 7 }
	};
	std::vector<std::vector<float>> ds{
		{ 7, 6, 5, 4, },
		{ 3, 2, 1, 9, },
		{ 8, 7, 6, 5, },
		{ 4, 3, 2, 1 }
	};
	std::vector<float> cs{ 8, 7, 6, 5 };*/

	auto e = Vector<float>(1000, randomizer).toStd();
	auto f = e;
	int i = 0;
	for (; i < 1e5; i++) {
		f = Vector<float>(f).toStd();
		//float s = 3;
		//auto rs = (b * d).getData().toStd();// - c;
		//auto rs = a.toStd();

		//bs = mul(bs, ds);//forAll(as, cs, op);
		equal(f, f);
		break;
	}
	

	
	//ThreadPool::finish();
	//exit(0);
	std::vector<Sample<float>> samples{
		Sample<float>(Vector<float>{ {0.0f, 0.0f, 0.0f} },Vector<float>{ 0.0f }),
		Sample<float>(Vector<float>{ {1.0f, 1.0f, 1.0f} },Vector<float>{ 1.0f }),
		Sample<float>(Vector<float>{ {0.0f, 1.0f, 0.0f} },Vector<float>{ 1.0f }),
		Sample<float>(Vector<float>{ {0.0f, 1.0f, 1.0f} },Vector<float>{ 1.0f }),
		Sample<float>(Vector<float>{ {0.0f, 0.0f, 0.0f} },Vector<float>{ 0.0f }),
		Sample<float>(Vector<float>{ {1.0f, 0.0f, 1.0f} },Vector<float>{ 1.0f }),
		Sample<float>(Vector<float>{ {1.0f, 1.0f, 0.0f} },Vector<float>{ 1.0f })
	};



	ANN network({
		3,
		512 * 512,// * 25,
		//128 * 128,// * 25,
		1
	});

	std::cout << "1.0 1.0 1.0: " << network.feedForward(std::vector<float>{1.0, 1.0, 1.0})[0] << std::endl;
	std::cout << "0.0 1.0 1.0: " << network.feedForward(std::vector<float>{0.0, 1.0, 1.0})[0] << std::endl;
	std::cout << "1.0 1.0 0.0: " << network.feedForward(std::vector<float>{1.0, 1.0, 0.0})[0] << std::endl;
	std::cout << "1.0 0.0 1.0: " << network.feedForward(std::vector<float>{1.0, 0.0, 1.0})[0] << std::endl;
	std::cout << "0.0 0.0 0.0: " << network.feedForward(std::vector<float>{0.0, 0.0, 0.0})[0] << std::endl;

	for (int i = 0; i < 20000; ++i) {
		network.learn(0.00001f, samples);
		if (i % 1 == 0) {
			std::cout << std::endl;
			std::cout << std::endl;
			std::cout << "1.0 1.0 1.0: " << network.feedForward(std::vector<float>{1.0, 1.0, 1.0})[0] << std::endl;
			std::cout << "0.0 1.0 1.0: " << network.feedForward(std::vector<float>{0.0, 1.0, 1.0})[0] << std::endl;
			std::cout << "1.0 1.0 0.0: " << network.feedForward(std::vector<float>{1.0, 1.0, 0.0})[0] << std::endl;
			std::cout << "1.0 0.0 1.0: " << network.feedForward(std::vector<float>{1.0, 0.0, 1.0})[0] << std::endl;
			std::cout << "0.0 0.0 0.0: " << network.feedForward(std::vector<float>{0.0, 0.0, 0.0})[0] << std::endl;
		}
	}
	//system("pause");
	return 0;
}