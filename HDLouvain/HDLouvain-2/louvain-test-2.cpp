#include <iostream>
#include <fstream>
#include <map>
#include <unordered_map>
#include <numeric>

#include "struct.h"
#include "louvain.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <iomanip>

//using namespace std;
//namespace py = pybind11;



int run_louvain_from_csv(const std::string &filename, int k = 15) {

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


    static adjlist adj;
    static  Louvain_Partition p;

    // Smallest distances
for (size_t i = 0; i < matrix.size(); i++) {
    std::vector<std::pair<double, int>> values;

    for (size_t j = 0; j < matrix[i].size(); j++)
        values.push_back({matrix[i][j], static_cast<int>(j)});  // (value, column_index)

    // Sort ascending by value explicitly
    sort(values.begin(), values.end(),
        [](const std::pair<double,int> &a, const std::pair<double,int> &b) {
            return a.first < b.first;
        }
    );

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

 graph_process(adj, p);
 louvain_main(p);
 std::cout << "louvain main done" << std::endl;

std::cout << std::endl;
//Print Labels: 
for (auto &entry : p.labels) {
        for (int v : entry.second) {
           std::cout << v << " " << entry.first << std::endl;
        }
}

    return 0;


   }



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./louvain_test <csv_file> [k]\n";
        return 1;
    }

    std::string filename = argv[1];
    int k = (argc > 2) ? std::stoi(argv[2]) : 15;

    try {
         run_louvain_from_csv(filename, k);

/*       std::cout << "\n=== Louvain Result ===\n";
        for (const auto &entry : result) {
            std::cout << "Community " << entry.first << ": ";
            for (int v : entry.second)
                std::cout << v << " ";
            std::cout << "\n";
        }*/
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
