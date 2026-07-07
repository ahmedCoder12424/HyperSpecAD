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


using namespace std;

int main(int argc, char* argv[]) {

//   adjlist adj;
//     Louvain_Partition p;

//     if (argc < 2) {
//         cerr << "Usage: " << argv[0] << " test.txt" << endl;
//         return 1;
//     }

//     string filename = argv[1];
//     ifstream infile(filename);
//     if (!infile) {
//         cerr << "Error opening file: " << filename << endl;
//         return 1;
//     }

//     int u, v;

//     while (infile >> u >> v) {
//         adj.graph[u][v] = 1;
//         adj.graph[v][u] = 1; // undirected edge
//         p.m++;   
//     }
//      p.m=p.m*2;
//     infile.close();


    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input.csv>" << endl;
        return 1;
    }

    string filename = argv[1];
    const int k = 15;  // Always pick 15 smallest values per row

    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return 1;
    }

    vector<vector<double>> matrix;
    string line;

    // --- Read CSV file ---
    while (getline(fin, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ',')) {  // use comma for CSV
            if (!value.empty())
                row.push_back(stod(value));
        }

        if (!row.empty())
            matrix.push_back(row);
    }
    fin.close();

    cout << fixed << setprecision(6);
    
     adjlist adj;
     Louvain_Partition p; 

    // Smallest distances
for (size_t i = 0; i < matrix.size(); i++) {
    vector<pair<double, int>> values;

    for (size_t j = 0; j < matrix[i].size(); j++)
        values.push_back({matrix[i][j], static_cast<int>(j)});  // (value, column_index)

    // Sort ascending by value explicitly
    sort(values.begin(), values.end(),
        [](const pair<double,int> &a, const pair<double,int> &b) {
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
        vector<pair<double, int>> values;

        for (size_t j = 0; j < matrix[i].size(); j++)
            values.push_back({matrix[i][j], static_cast<int>(j)}); // (value, index)

        sort(values.begin(), values.end(),
             [](const pair<double,int>& a, const pair<double,int>& b) {
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
    //cout << adj.graph.size() << " " << p.m << endl;

    // --- Print graph adjacency list ---
//    for (auto& node : adj.graph) {
//     cout << "Node " << node.first << ": ";
//     for (auto& neighbor : node.second) {
//         cout << neighbor.first << " "; // only neighbor index
//     }
//     cout << endl;
// }

    p.l=adj.graph.size();
    for (const auto& [u, neighbors] : adj.graph) { p.ed+=adj.graph[u].size();}
    cout << p.l << " " << p.ed << endl;        
                                                                                                                                           
 graph_process(adj, p);
 louvain_main(p);


 cout << endl;
//Print Labels: 
for (auto &entry : p.labels) {
        for (int v : entry.second) {
           cout << v << " " << entry.first << endl;
        }
}

    return 0;
}
