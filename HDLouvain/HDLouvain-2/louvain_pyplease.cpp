#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>

#include "struct.h"
#include "louvain.h"

namespace py = pybind11;

   std::vector<int>  run_louvain_from_csv(const std::string &filename, int k = 15) {

    std::ifstream fin(filename);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }

    std::vector<std::vector<double>> matrix;
    std::string line;
    while (getline(fin, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (getline(ss, value, ',')) {
            if (!value.empty())
                row.push_back(std::stod(value));
        }
        if (!row.empty())
            matrix.push_back(row);
    }
    fin.close();

    static adjlist adj;
    static Louvain_Partition p;

    // Build graph
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<std::pair<double,int>> values;
        for (size_t j = 0; j < matrix[i].size(); j++)
            values.push_back({matrix[i][j], static_cast<int>(j)});
        std::sort(values.begin(), values.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });

        for (int t = 0; t < k && t < static_cast<int>(values.size()); t++) {
            int u = i;
            int v = values[t].second;
            adj.graph[u][v] = 1;
            adj.graph[v][u] = 1;
            p.m++;
        }
    }
    p.m = 2 * p.m;

    p.l = adj.graph.size();
    p.ed = 0;
    for (const auto& [u, neighbors] : adj.graph)
        p.ed += neighbors.size();

    graph_process(adj, p);
    louvain_main(p);

    // Convert labels map to Python dict
  //  std::map<int, std::vector<int>> result = p.labels;
/**    for (const auto& entry : result) {
    std::cout << "Community " << entry.first << ": ";
    for (int node : entry.second) {
        std::cout << node << " ";
    }
    std::cout << std::endl;

}**/
//
//    std::vector<std::pair<int,int>> result;
    std::vector<int> result;
    for (auto &entry : p.labels) {
        for (int v : entry.second) {
	//	std::cout << v << " " << entry.first << std::endl;
		result.push_back(entry.first);
        }
}



    return result;
}

// Pybind11 module
PYBIND11_MODULE(louvain_py, m) {
    m.doc() = "HDLouvain module via pybind11";
    m.def("run_louvain_from_csv", &run_louvain_from_csv,
          py::arg("filename"), py::arg("k") = 15,
          "Run Louvain clustering from a CSV file");
}

