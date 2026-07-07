
#include "struct.h"
#include <numeric> 
#include <execution>

#include "louvain.h"

using namespace std;


//Louvain_Partition
void graph_process (adjlist& adj, Louvain_Partition& p)
{


for (int i = 0; i <p.l; i++) {
p.labels[i].push_back(i);  // put one value in each vector
}

// for (auto &entry : p.labels) {
//     std::cout << entry.first << ": ";
//     for (int v : entry.second)
//         std::cout << v << " ";
//     std::cout << std::endl;
// }

p.l = adj.graph.size();    
/*
p.e =  new int [p.ed+p.l];
p.wts   =  new double [p.ed+p.l]; 
p.ctr =    new int [(p.l+1)];
p.in   =   new double [p.l];          
p.d    =   new double [p.l];           
p.td   =   new double [p.l];         
p.slps =   new double [p.l];  
p.real_id=  new int [p.l];      
p.nc   =   new int [p.l];  
*/

p.e.resize(p.ed + p.l);
p.wts.resize(p.ed + p.l);
p.ctr.resize(p.l + 1);
p.in.resize(p.l);
p.d.resize(p.l);
p.td.resize(p.l);
p.slps.resize(p.l);
p.real_id.resize(p.l);
p.nc.resize(p.l);


/*
std::fill( p.in, p.in+p.l, 0.0);
std::fill( p.d, p.d+p.l, 0.0);
std::fill( p.td,  p.td+p.l, 0.0);
std::fill( p.slps, p.slps+p.l, 0.0);
*/

std::fill(p.in.begin(),   p.in.end(),   0.0);
std::fill(p.d.begin(),    p.d.end(),    0.0);
std::fill(p.td.begin(),   p.td.end(),   0.0);
std::fill(p.slps.begin(), p.slps.end(), 0.0);

p.weight=0;
for (int i = 0; i < p.l; i++)
{
    p.nc[i] = i;
    p.real_id[i]=i;
}

for (auto& pair : adj.graph) {
    int node = pair.first;
    auto& neighbors = pair.second;

    p.d[node] = p.td[node] = neighbors.size();  

    p.weight+= p.d[node];
    for (const auto& [v, _] : neighbors) {
        if (node == v) {
            p.slps[node] = 1;  
        }
    }
}

int idx = 0;
int inc=1;
p.ctr[0] = idx;

for (auto& pair : adj.graph) {
    int u = pair.first;
    auto& neighbors = pair.second;

    neighbors[u] = u;

    for (const auto& [v, _] : neighbors) {
     p.e[idx++] = v;
     p.wts[idx]=1;

    }
    p.ctr[inc] = idx;
    inc++;
}
// cout << inc << endl;
// cout << endl;
//  for (int i = 0; i < 61; i++) { 
//     cout << p.e[i] << " " ;
//  }

//  for (int i = 0; i < 15; i++) { 
//     p.wts[p.ctr[i]]=0;
//  }

  for (int i = 0; i <p.l; ++i) {                    // this is extra work  (67-70) we actually don't need it
        int start = p.ctr[i]+1;
        int end = p.ctr[i + 1];
        //std::sort(p.e + start, p.e + end);    }
	std::sort(p.e.begin() + start, p.e.begin() + end);}
  //  return std::move(p);
  // return p;
}




