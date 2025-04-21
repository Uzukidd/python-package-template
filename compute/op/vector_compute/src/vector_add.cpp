#include <torch/extension.h>

#include "vector_add.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add_cuda", &vector_add_cuda, "Vector addition (CUDA)");
}