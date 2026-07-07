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



struct adjlist {
    std::map<int, std::unordered_map<int, int>> graph;
};

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


Louvain_Partition graph_process (adjlist& adj, Louvain_Partition& p);
double modularity(Louvain_Partition& p) ;
#endif // STRUCT_H





