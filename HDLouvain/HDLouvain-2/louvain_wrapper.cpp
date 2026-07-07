#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "struct.h"
#include "louvain.h"
#include "struct.h"

using namespace std;

// Define a callable function instead of main()
void run_louvain(const std::string &filename, int k = 15) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }

    vector<vector<double>> matrix;
    string line;

    // --- Read CSV file ---
    while (getline(fin, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ',')) {
            if (!value.empty())
                row.push_back(stod(value));
        }

        if (!row.empty())
            matrix.push_back(row);
    }
    fin.close();

    adjlist adj;
    Louvain_Partition p;

    // Build undirected graph
    for (size_t i = 0; i < matrix.size(); i++) {
        vector<pair<double, int>> values;
        for (size_t j = 0; j < matrix[i].size(); j++)
            values.push_back({matrix[i][j], static_cast<int>(j)});

        sort(values.begin(), values.end(),
            [](const pair<double,int>& a, const pair<double,int>& b) {
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

    cout << "Graph: " << p.l << " nodes, " << p.ed << " edges" << endl;

    graph_process(adj, p);
    louvain_main(p);

    // Print Labels
    for (auto &entry : p.labels) {
        for (int v : entry.second) {
            cout << v << " " << entry.first << endl;
        }
    }
}

