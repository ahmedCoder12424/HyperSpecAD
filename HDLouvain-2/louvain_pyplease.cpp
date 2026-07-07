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
#include <climits> 
namespace py = pybind11;

   std::vector<int>  run_louvain(const std::vector<std::vector<double>> &matrix,  int k) {



//	   std::cout << "Matrix size: " << matrix.size() << " rows" << std::endl;
/*f (!matrix.empty()) {
    //std::cout << "Matrix[0] size: " << matrix[0].size() << " cols" << std::endl;
}

for (size_t i = 0; i < std::min<size_t>(matrix.size(), 3); ++i) {
    std::cout << "Row " << i << " (" << matrix[i].size() << " cols): ";
    for (size_t j = 0; j < std::min<size_t>(matrix[i].size(), 10); ++j) {
  //      std::cout << matrix[i][j] << " ";
    }
// std::cout << std::endl;
}*/
    if (k == -1){
        k = INT_MAX;
    }
    adjlist adj;
    Louvain_Partition p;

    // Build graph
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<std::pair<double,int>> values;
        for (size_t j = 0; j < matrix[i].size(); j++)
            values.push_back({matrix[i][j], static_cast<int>(j)});
        std::sort(values.begin(), values.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });

        for (int t = 0; t < k &&  t < static_cast<int>(values.size()); t++) {
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
//
    
  //  std::cout << "SIZE OF P.LABELS " << p.labels.size() << std::endl;
    std::vector<int> result;
    size_t total_points = 0;
    for (auto &entry : p.labels) {
//	   std::cout << "Cluster " << entry.first << " has " << entry.second.size() << " points" << std::endl;
        for (int v : entry.second) {
//std::cout << v << " " << entry.first << std::endl;
		result.push_back(entry.first);
		total_points +=1;
        }
}

  //  std::cout << "Total data points: " << total_points << std::endl;
  //  std::cout << "Result vector size: " << result.size() << std::endl;

    return result;
}

// Pybind11 module
PYBIND11_MODULE(louvain_py, m) {
    m.doc() = "HDLouvain module via pybind11";
    m.def("run_louvain", &run_louvain,
          py::arg("matrix"), py::arg("k"),
          "Run Louvain clustering from a CSV file");
}

