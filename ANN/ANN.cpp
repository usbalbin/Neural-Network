// ANN.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ANN.hpp"


//#include "Matrix.hpp"
#include "Utils.hpp"
#include "Math.hpp"
#include <iostream>
#include <numeric> //TODO: remove me





ANN::ANN(const std::vector<size_t>& layerSizes){
	//std::random_device randomDevice = std::random_device();
	std::mt19937 randomEngine = std::mt19937(0/*randomDevice()*/);
	std::normal_distribution<double> distribution(0, 1);
	std::function<double(void)> distributor = std::bind(distribution, randomEngine);
	

	for (size_t i = 1; i < layerSizes.size(); ++i) {
		layers.emplace_back(layerSizes[i - 1], layerSizes[i], distributor);
	}
}

std::vector<float> ANN::feedForward(std::vector<float>& input)
{
	assert(input.size() == layers[0].weights.getRowCount());

	Vector<float> data = input;
	for (auto& layer : layers)
		data = activationFunc(data * layer.weights + layer.biases);

	return data.toStd();
}

void ANN::backPropogate(Sample<float>& sample, std::vector<Matrix<float>>& gradientsWeightsOut, std::vector<Vector<float>>& gradientsBiasesOut, bool initOutputs)
{
	if (initOutputs) {
		gradientsBiasesOut.resize(layers.size());
		gradientsWeightsOut.resize(layers.size());

		for (size_t i = 0; i < layers.size(); ++i) {
			gradientsBiasesOut[i] = Vector<float>(layers[i].biases.size(), 0);
			gradientsWeightsOut[i] = Matrix<float>(layers[i].weights.getRowCount(), layers[i].weights.getColumnCount(), 0);
		}
	}
	assert(gradientsBiasesOut.size() == layers.size() && gradientsWeightsOut.size() == layers.size());

	auto activation = sample.indata;
	//auto remove = sample.indata.toStd();
	//auto removeToo = sample.expectedResult.toStd();
	
	std::vector<Vector<float>> rawData;		//Every layers raw output(without activationFunc)
	std::vector<Vector<float>> activations;	//Every layers activations(raw with activationFunc applied)


	//Feedforward
	for (auto& layer : layers) {
		auto raw = activation * layer.weights + layer.biases;
		//auto removeMe = raw.toStd();					//TODO: remove

		rawData.push_back(raw);

		activation = activationFunc(raw);
		activations.push_back(activation);

		//auto removeMeToo = activation.toStd();					//TODO: remove
	}

	//Work backwards to calculate deltas
	Vector<float> delta;
	
	{
		auto& out = activations[activations.size() - 1];
		auto& in = (activations.size() > 1) ? activations[activations.size() - 2] : sample.indata;

		delta = (out - sample.expectedResult) * activationFuncPrime(rawData.back());
		//auto removeMePlz = delta.toStd();							//TODO: remove me
		gradientsBiasesOut.back() += delta;
		gradientsWeightsOut.back() += mulColumnRow(in, delta);//TODO: Check if "in" and "delta" should be swapped
	}

	for (int i = layers.size() - 2; i > 0; i--) {
		auto& out = activations[i - 0];
		auto& in = activations[i - 1];

		delta = mulTranspMatrix(delta, layers[i + 1].weights) * activationFuncPrime(rawData[i]);
		//auto removeMeNow = delta.toStd();							//TODO: remove me
		gradientsBiasesOut[i] += delta;
		gradientsWeightsOut[i] += mulColumnRow(in, delta);//TODO: Check if "in" and "delta" should be swapped
	}

	if (activations.size() > 1)
	{
		auto& out = activations[0];
		auto& in = sample.indata;

		delta = mulTranspMatrix(delta, layers[1].weights) * activationFuncPrime(rawData[0]);
		gradientsBiasesOut[0] += delta;
		gradientsWeightsOut[0] += mulColumnRow(in, delta);//TODO: Check if "in" and "delta" should be swapped
	}
}

void ANN::applyGradients(float learningRate, std::vector<Matrix<float>>& gradientsWeights, std::vector<Vector<float>>& gradientsBiases)
{
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i].biases -= learningRate * gradientsBiases[i];

		layers[i].weights -= learningRate * gradientsWeights[i];
	}
}

/*void ANN::learn(float learningRate, Sample<float>& sample)
{
	learn(learningRate, { sample });
}*/

void ANN::learn(float learningRate, std::vector<Sample<float>>& samples)
{
	bool init = true;
	std::vector<Vector<float>> gradientBiases;
	std::vector<Matrix<float>> gradientWeights;
	for (auto& sample : samples) {
		backPropogate(sample, gradientWeights, gradientBiases, init);
		init = false;
	}

	applyGradients(learningRate, gradientWeights, gradientBiases);
}