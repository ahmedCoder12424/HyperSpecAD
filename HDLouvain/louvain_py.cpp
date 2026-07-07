#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "struct.h"
#include "louvain.h"

namespace py = pybind11;

//std::vector<int>
   std::map<int, std::vector<int>>run_louvain_from_csv(const std::string &filename, int k = 15) {
    // Read CSV
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

    std::cout << "done parsing" << std::endl;
    // Build adjacency graph and partition
    adjlist adj;
    Louvain_Partition p;

   // p.m = 0;
  //  p.l = matrix.size();
/*
    // Construct graph: connect each node to k smallest values
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<std::pair<double,int>> values;
        for (size_t j = 0; j < matrix[i].size(); j++)
            values.push_back({matrix[i][j], static_cast<int>(j)});
        
        std::sort(values.begin(), values.end(),
                  [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                      return a.first < b.first;
                  });

        for (int t = 0; t < k && t < static_cast<int>(values.size()); t++) {
            int u = i;
            int v = values[t].second;
            adj.graph[u][v] = 1;
            adj.graph[v][u] = 1;
            p.m++;
        }
    }
    p.m *= 2; // same as your main.cpp logic
*/

    for (size_t i = 0; i < matrix.size(); i++) {
	std::vector<std::pair<double, int>> values;

    for (size_t j = 0; j < matrix[i].size(); j++)
        values.push_back({matrix[i][j], static_cast<int>(j)});  // (value, column_index)

    // Sort ascending by value explicitly
    std::sort(values.begin(), values.end(),
        [](const std::pair<double,int> &a, const std::pair<double,int> &b) {
            return a.first < b.first;
        }
    );

    // cout << "Row " << i << " -> ";
    // for (int t = 0; t < k && t < static_cast<int>(values.size()); t++) {
    //     cout << "(" << values[t].second << ", " << fixed << setprecision(6) << values[t].first << ") ";
    // }
    // cout << endl;
}

    // --- Build undirected graph ---
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<std::pair<double, int>> values;

        for (size_t j = 0; j < matrix[i].size(); j++)
            values.push_back({matrix[i][j], static_cast<int>(j)}); // (value, index)

        sort(values.begin(), values.end(),
             [](const std::pair<double,int>& a, const std::pair<double,int>& b) {
                 return a.first < b.first;
             });

        for (int t = 0; t < k && t < static_cast<int>(values.size()); t++) {
            int u = i;
            int v = values[t].second;
            double s = values[t].first;
            // Add undirected edge
            adj.graph[u][v] = 1;
            adj.graph[v][u] = 1; // undirected edge
            p.m++;
             //cout << i << " "<< v << " " << s << endl;
        }
    }
    p.m=2*p.m;

      p.l=adj.graph.size();
    for (const auto& [u, neighbors] : adj.graph) { p.ed+=adj.graph[u].size();}
    std::cout << p.l << " " << p.ed << std::endl;

        std::cout << "done building graph" << std::endl;
 graph_process(adj, p);

        std::cout << "done processing graph" << std::endl;
 louvain_main(p);

        std::cout << "done preforming louvain" << std::endl;

 std::cout << std::endl;
//Print Labels:
/*for (auto &entry : p.labels) {
        for (int v : entry.second) {
		std::cout << v << " " << entry.first << std::endl;
        }
}
return p.labels;*/

  std::map<int, std::vector<int>> result_labels;

    for (auto &entry : p.labels) {
        result_labels[entry.first] = entry.second; // This creates copies of the vectors
    }
/*
   p.e = nullptr;
    p.ctr = nullptr;
    p.in = nullptr;
    p.d = nullptr;
    p.td = nullptr;
    p.slps = nullptr;
    p.wts = nullptr;
    p.nc = nullptr;
    p.real_id = nullptr;
    p.labels.clear();*/
    return result_labels; // Return the new copy

   /* // Count edges
    for (const auto &node : adj.graph)
        p.ed += node.second.size();

    // Process graph into partition
    graph_process(adj, p);

     std::cout << "graph process done" << std::end;
    // Run Louvain
    louvain_main(p);
    std::cout << "Louvain done" << std::end;
    std::vector<int> safe_labels(p.l, -1);
    for (auto &entry : p.labels) {
    	int comm = entry.first;
    	for (int node : entry.second)
        	if (node >=0 && node < (int)safe_labels.size())
            		safe_labels[node] = comm;
   	}	
     
// Clear internal memory that might get freed twice
     p.labels.clear();
p.nc = nullptr; // if p.nc is a raw pointer

	return safe_labels;

     */                  
   //std::vector<int> labels;
  //  labels.reserve(p.l);
  //  labels.assign(p.l, -1);
    // Return final community labels as vector<int>
    //td::vector<int> labels(p.l);
  //  for (int i = 0; i < p.l; i++)
    //    labels[i] = p.nc[i];
  //  std::vector<int> labels(p.l);
  //  for (const auto &entry : p.labels) {
  //  int comm = entry.first;
   // for (int node : entry.second) {
     //   if (node >= 0 && node < (int)labels.size())
       //     labels[node] = comm;
    //  }
   // }
   // return labels;
}


PYBIND11_MODULE(louvain_module, m) {
    m.def("run_louvain", &run_louvain_from_csv, py::arg("filename"), py::arg("k") = 15,
          "Run Louvain clustering from a CSV file and return community labels");
}

