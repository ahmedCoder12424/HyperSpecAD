// louvain_python.cpp
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "louvain.h"
#include "struct.h"
#include <vector>
#include <algorithm>
#include <map>

// Helper: Convert numpy array to std::vector
std::vector<std::vector<double>> numpy_to_vector(PyArrayObject* array) {
    int n_rows = PyArray_DIM(array, 0);
    int n_cols = PyArray_DIM(array, 1);
    double* data = static_cast<double*>(PyArray_DATA(array));
    std::vector<std::vector<double>> mat(n_rows, std::vector<double>(n_cols));
    for (int i = 0; i < n_rows; i++)
        for (int j = 0; j < n_cols; j++)
            mat[i][j] = data[i*n_cols + j];
    return mat;
}

// Python wrapper for Louvain
static PyObject* py_run_louvain(PyObject* self, PyObject* args) {
    PyObject* input_obj;
    int k;

    if (!PyArg_ParseTuple(args, "Oi", &input_obj, &k))
        return NULL;

    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(input_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!array) return NULL;

    auto matrix = numpy_to_vector(array);
    Py_DECREF(array);

    adjlist adj;
    Louvain_Partition p;

    // Build k-smallest graph
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<std::pair<double,int>> values;
        for (size_t j = 0; j < matrix[i].size(); j++)
            values.push_back({matrix[i][j], (int)j});
        std::sort(values.begin(), values.end(), [](auto &a, auto &b){ return a.first < b.first; });

        for (int t = 0; t < k && t < (int)values.size(); t++) {
            int u = i;
            int v = values[t].second;
            adj.graph[u][v] = 1;
            adj.graph[v][u] = 1;
            p.m++;
        }
    }
    p.m *= 2;
    p.l = adj.graph.size();
    for (const auto& [u, neighbors] : adj.graph)
        p.ed += neighbors.size();

    graph_process(adj, p);
      std::cout << "louvain main starting" << std::endl;
    louvain_main(p);
     
    std::cout << "louvain main done" << std::endl;
    // Return labels as numpy array
    npy_intp dims[1] = {p.l};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_INT32);
    int* out_ptr = static_cast<int*>(PyArray_DATA((PyArrayObject*)result));

    for (int node = 0; node < p.l; node++) {
        if (p.labels.find(node) != p.labels.end())
            out_ptr[node] = p.labels[node][0];
        else
            out_ptr[node] = -1;
    }

    return result;
}

// Module method table
static PyMethodDef LouvainMethods[] = {
    {"run_louvain", py_run_louvain, METH_VARARGS, "Run Louvain clustering on a distance matrix"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef louvainmodule = {
    PyModuleDef_HEAD_INIT,
    "louvain_module",  // module name
    NULL,
    -1,
    LouvainMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_louvain_module(void) {
    import_array(); // Required for numpy
    return PyModule_Create(&louvainmodule);
}

