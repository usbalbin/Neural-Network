#pragma once

#include <vector>
#include "Vector.hpp"
#include "Matrix.hpp"


template<typename T>
struct Sample {
	Sample(Vector<T>& indata, Vector<T>& expectedResult) : indata(indata), expectedResult(expectedResult){}
	Vector<T> indata;
	Vector<T> expectedResult;
};

template<typename T>
struct Layer {
	Layer(size_t inputCount, size_t nodeCount, std::function<double(void)>& distributor)
	{
		double factor = 1.0 / std::sqrt(inputCount);
		std::function<double(void)> initializer = [&distributor, factor] { return factor * distributor();  };

		weights = Matrix<T>(inputCount, nodeCount, initializer);
		biases = Vector<T>(nodeCount, distributor);
	};
	Vector<T> biases;
	Matrix<T> weights;
};

#define activationFunc sigmoid
#define activationFuncPrime sigmoidPrime

class ANN {
public:
	/*
	example
	{ 
		2	input layer
		4	hidden layer(s)
		3	hidden layer(s)
		1	output layer
	}
	
	*/
	ANN(const std::vector<size_t>& layerSizes);

	std::vector<float> feedForward(std::vector<float>& input);

	/*Requires outputs to be resized to proper sizes
	  outputs will be incremented, thus you will have to zero them first time.
	*/
	inline void backPropogate(
		Sample<float>& sample, 
		std::vector<Matrix<float>>& gradientsWeightsOut, std::vector<Vector<float>>& gradientsBiasesOut,
		bool initOutputs=true
	);

	void applyGradients(float learningRate, std::vector<Matrix<float>>& gradientsWeights, std::vector<Vector<float>>& gradientsBiases);

	//inline void learn(float learningRate, Sample<float>& sample);
	void learn(float learningRate, std::vector<Sample<float>>& samples);

private:
	std::vector<Layer<float>> layers;
};