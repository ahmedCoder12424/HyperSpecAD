#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <algorithm>

#include "struct.h"
#include "louvain.h"

// This version is identical to the pybind version but has a main()
// so you can test from C++ directly.
std::map<int, std::vector<int>> run_louvain_from_csv(const std::string &filename, int k = 15) {
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

    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<std::pair<double, int>> values;
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

    p.m = 2 * p.m;
    p.l = adj.graph.size();
    for (const auto& [u, neighbors] : adj.graph)
        p.ed += adj.graph[u].size();

    std::cout << p.l << " " << p.ed << std::endl;
    std::cout << "done building graph" << std::endl;

    graph_process(adj, p);
    std::cout << "done processing graph" << std::endl;

    louvain_main(p);
    std::cout << "done performing louvain" << std::endl;

    std::map<int, std::vector<int>> result_labels;
    for (auto &entry : p.labels) {
        result_labels[entry.first] = entry.second;
    }

    return result_labels;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./louvain_test <csv_file> [k]\n";
        return 1;
    }

    std::string filename = argv[1];
    int k = (argc > 2) ? std::stoi(argv[2]) : 15;

    try {
        auto result = run_louvain_from_csv(filename, k);

        std::cout << "\n=== Louvain Result ===\n";
        for (const auto &entry : result) {
            std::cout << "Community " << entry.first << ": ";
            for (int v : entry.second)
                std::cout << v << " ";
            std::cout << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

