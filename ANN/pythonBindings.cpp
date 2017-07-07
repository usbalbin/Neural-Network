#include "pythonBindings.hpp"

#ifdef PYTHON_BINDING


#include <pybind11/pybind11.h>
#include <pybind11\stl.h>
#include "ANN.hpp"
#include "Vector.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ANN, m) {
	py::class_<ANN>(m, "ANN")
		.def(py::init<const std::vector<size_t>&>())
		.def(py::init<std::string&>())
		.def("feedForward", &ANN::feedForward)
		.def("learn", &ANN::learn)
		.def("writeToFile", &ANN::writeToFile);

	py::class_<Vector<float>>(m, "Vector")
		.def(py::init<std::vector<float>&>());

	py::class_<Sample<float>>(m, "Sample")
		.def(py::init<Vector<float>&, Vector<float>&>());

	m.def("initCL", &VectorOps::init, "Initialize OpenCL",
		py::arg("deviceType") = CL_DEVICE_TYPE_GPU);
}










#endif // PYTHON_BINDING