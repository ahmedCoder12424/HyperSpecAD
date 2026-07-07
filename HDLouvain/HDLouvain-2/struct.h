#ifndef STRUCT_H
#define STRUCT_H

#include <unordered_map>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <map>
#include <chrono>

#include <memory>

struct adjlist {
    std::map<int, std::unordered_map<int, int>> graph;
};


/*
struct Louvain_Partition {
    int* e = nullptr;           // CSR neighbours
    int* ctr = nullptr;         // neighbour's count
    double* in = nullptr;       // nodes inside one community
    double* d = nullptr;        // degree
    double* td = nullptr;       // running sum of degree
    double* slps = nullptr;     // self_loops
    double* wts = nullptr;      // weights
    int* nc = nullptr;          // node community
    int* real_id = nullptr;     // real community IDs
    
    double weight = 0;
    int ed = 0;
    int l = 0;
    double m = 0;  
    std::map<int, std::vector<int>> labels;

    Louvain_Partition() = default;
    // Simple destructor
    ~Louvain_Partition() {
        delete[] e;
        delete[] ctr;
        delete[] in;
        delete[] d;
        delete[] td;
        delete[] slps;
        delete[] wts;
        delete[] nc;
        delete[] real_id;
    }
    
    // Prevent copying to avoid double-free
    Louvain_Partition(const Louvain_Partition&) = delete;
    Louvain_Partition& operator=(const Louvain_Partition&) = delete;
    
    // Allow moving (optional but good practice)
    Louvain_Partition(Louvain_Partition&& other) noexcept {
        *this = std::move(other);
    }
    
    Louvain_Partition& operator=(Louvain_Partition&& other) noexcept {
        if (this != &other) {
            // Delete our current memory
            delete[] e; delete[] ctr; delete[] in; delete[] d;
            delete[] td; delete[] slps; delete[] wts; delete[] nc; delete[] real_id;
            
            // Take ownership of other's memory
            e = other.e; other.e = nullptr;
            ctr = other.ctr; other.ctr = nullptr;
            in = other.in; other.in = nullptr;
            d = other.d; other.d = nullptr;
            td = other.td; other.td = nullptr;
            slps = other.slps; other.slps = nullptr;
            wts = other.wts; other.wts = nullptr;
            nc = other.nc; other.nc = nullptr;
            real_id = other.real_id; other.real_id = nullptr;
            
            // Copy other members
            weight = other.weight;
            ed = other.ed;
            l = other.l;
            m = other.m;
            labels = std::move(other.labels);
        }
        return *this;
    }
};
*/




struct Louvain_Partition {
    std::vector<int> e;          // concatenated list of neighbours in CSR format
   std::vector<int> ctr;           // neighbour's count
    std::vector<double> in;    // nodes inside one community
    std::vector<double> d;      // degree
    std::vector<double> td;
          // running sum of degree
     std::vector<double> wts;
     std::vector<double> slps;         // self_loops     // weights
       std::vector<int> nc;         // node community where node belongs
    double weight;
    int ed;           // total edges in the graph
    int l;              // total vertices in the graph
    double m;
    std::map<int, std::vector<int>> labels;
    std::vector<int> real_id;
};


/*
struct Louvain_Partition {
    int* e;           // concatenated list of neighbours in CSR format
    int* ctr;         // neighbour's count
    double* in;          // nodes inside one community
    double* d;           // degree
    double* td;          // running sum of degree
    double* slps;        // self_loops
    double* wts;        // weights
    int* nc;          // node community where node belongs 
    double weight;
    int ed;           // total edges in the graph
    int l;              // total vertices in the graph
    double m;  
    std::map<int, std::vector<int>> labels;
    int* real_id;
};
*/
/*
struct Louvain_Partition {
    std::unique_ptr<int[]> e;           // CSR neighbours
    std::unique_ptr<int[]> ctr;         // neighbour's count
    std::unique_ptr<double[]> in;       // nodes inside one community
    std::unique_ptr<double[]> d;        // degree
    std::unique_ptr<double[]> td;       // running sum of degree
    std::unique_ptr<double[]> slps;     // self_loops
    std::unique_ptr<double[]> wts;      // weights
    std::unique_ptr<int[]> nc;          // node community
    std::unique_ptr<int[]> real_id;     // real community IDs
    
    double weight;
    int ed;
    int l;
    double m;  
    std::map<int, std::vector<int>> labels;

    Louvain_Partition() : e(nullptr), ctr(nullptr), in(nullptr), d(nullptr), 
                         td(nullptr), slps(nullptr), wts(nullptr), nc(nullptr),
                         real_id(nullptr), weight(0), ed(0), l(0), m(0) {}
    
    // No need for destructor - unique_ptr auto-deletes
};
*/


//Louvain_Partition 
void graph_process (adjlist& adj, Louvain_Partition& p);
double modularity(Louvain_Partition& p) ;
#endif // STRUCT_H





