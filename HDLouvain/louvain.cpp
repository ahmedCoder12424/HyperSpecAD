
#include "struct.h"
#include "louvain.h"
#include <iomanip> 
#include <cmath>

using namespace std;

int values=0;
int aggregate_graph(Louvain_Partition& p) 
{
    int* renumber = (int*) calloc(p.l, sizeof(int));
    int* count    = (int*) calloc(p.l+1, sizeof(int));          // can look into it; could be new size
    int* reorder = (int*) calloc(p.l, sizeof(int));
   
    for (int i = 0; i < p.l; i++) {
        renumber[i] = -1;
        reorder[i]=p.nc[i];
       
    }
    
    for (int i = 0; i < p.l; i++) {                                   //actual position+1 so that we can make CSR starting with 0; 
        count[p.nc[i]+1] += p.d[i];
    }

    int* idx    = (int*) calloc(p.l, sizeof(int)); 
    for (int i = 0; i < p.l; i++) idx[i] = i;

    sort(idx, idx+p.l, [&](int i, int j) {                              //this could be optimized by eliminating sorting and using index wise data re-ordering.
        return p.nc[i] < p.nc[j];
    });
    
      
    sort(reorder, reorder + p.l);
    unsigned long i, last = 0;
    int rem_clus = 0;
    for (i = 0; i < p.l; i++) {
        if (renumber[reorder[i]] == -1) {
            rem_clus++;
            renumber[reorder[i]] = last++;
        }
    }
   
   
    vector<map<int,double>> vec_of_maps(rem_clus);
    cout << "remaining is: " << rem_clus;
    for (int i = 0; i < p.l; i++) {
        for (int nbr = p.ctr[idx[i]]; nbr < p.ctr[idx[i]+1]; nbr++) {
            if (p.e[nbr] != idx[i]) {
                int q = renumber[p.nc[idx[i]]];
                int s = renumber[p.nc[p.e[nbr]]];
                vec_of_maps[q][s] += 1;
            }
        }
              
    }
    
    int cnt=0;
    for (int i = 0; i < vec_of_maps.size(); i++) {
    double sum=0;
    for (auto it = vec_of_maps[i].begin(); it != vec_of_maps[i].end(); ++it) {
        sum+=it->second;
        if(it->first==i)
        {
          p.in[i]=it->second;
          p.slps[i]=it->second;
        }
    }
       cnt+=vec_of_maps[i].size();
       p.ctr[i+1]=cnt;
       p.d[i]=sum;
       p.td[i]=sum;
   }
   cout << endl;
    p.ed=0;
    for (int i = 0; i < vec_of_maps.size(); i++) {
    for (auto it = vec_of_maps[i].begin(); it != vec_of_maps[i].end(); ++it) {
        p.e[p.ed]= it->first;
        p.wts[p.ed]= it->second;
        p.ed++;
    }
     p.nc[i]=i;
  }

                                                            // Printing
                                                            // for (int i = 0; i < vec_of_maps.size(); i++) {
                                                            //     std::cout << "Map " << i << ":\n";
                                                            //     for (auto it = vec_of_maps[i].begin(); it != vec_of_maps[i].end(); ++it) {
                                                            //         std::cout << "  Key: " << it->first << ", Value: " << it->second << "\n";
                                                            //     }
                                                            // }

                                                            // Printing


p.l=vec_of_maps.size();
louvain_main(p);


}


double modularity(Louvain_Partition& p) 
{
  double q = 0.0;
  for (int i = 0; i < p.l; i++) {
        if (p.td[i] > 0.0) {
            q += p.in[i] - (p.td[i] * p.td[i]) / (p.m + 0.0);
        }
    }
    return q / (p.m + 0.0);
}


int louvain_main (Louvain_Partition& p)
{

double quality=0.0;
double newGain=0.0;
int best_comm, old_comm,mvs;
double imp=0.0;
double prev_quality=0.0;
double q_prev_it=0;
double dncomm;
double val;

values ++;
quality= modularity(p);
cout << "old quality: " << setprecision(6) << quality << endl;
q_prev_it=quality; 

do {
  
prev_quality=quality;	
     mvs=0;
for (int i=0; i<p.l; i++)
{
 double bestGain=0.0;
 old_comm=p.nc[i];

double dnc=0.0;
for (int nbr=p.ctr[i]; nbr < p.ctr[i+1]; nbr++) 
 {
  if(p.e[nbr]!=i && p.nc[p.e[nbr]]==p.nc[i]) 
   {
     //dnc +=1;
     dnc +=p.wts[nbr];
   }
 }
 
//Remove node from its community
p.in[old_comm]-= 2 * dnc + p.slps[i];
p.td[old_comm]-= p.d[i];

for (int comm=p.ctr[i]; comm < p.ctr[i+1]; comm++) 
 {
    dncomm=0.0;
    int c=p.nc[p.e[comm]];
for (int nbr=p.ctr[i]; nbr<p.ctr[i+1]; nbr++) 
 {
  if(p.e[nbr]!=i && p.nc[p.e[nbr]] == c) 
   {
      //dncomm+=1;
      dncomm+=p.wts[nbr];
   }
 }

newGain = dncomm - (p.td[c] * p.d[i])/(p.m);

//if(newGain > bestGain)  
if(newGain > bestGain || (newGain==bestGain && c==old_comm))  
{ 
  bestGain=newGain;
  p.nc[i]=c;
  dnc=dncomm;  
}    
}        

//Add node to the community giving best gain: default is old community >>
p.in[p.nc[i]]+= 2 * dnc + p.slps[i];
p.td[p.nc[i]]+= p.d[i];

  if (p.nc[i] != old_comm) {
	mvs++;
      } 
}  

  quality= modularity(p);
  imp=quality-prev_quality;
  cout << "new quality: " << setprecision(6) << quality << "  imp  =" << quality-prev_quality  << endl;
  //printf("Quality is: %f\n", quality);
  cout << endl;
}
 while( mvs > 0 && imp > 0.005);
 
// Labeling fr each level
int index=0;
 for (auto &entry : p.labels) {
    if (index != p.nc[index] ) {
        for (int v : entry.second) {
           p.labels[p.real_id[p.nc[index]]].push_back(v);
        }
                entry.second.clear();
    }
    index++;
}

for (auto it = p.labels.begin(); it != p.labels.end(); ) {
    if (it->second.empty()) { // check if vector is empty
        it = p.labels.erase(it); 
    } else {
        ++it;
    }
}

 int idx = 0;
 for (auto &lab : p.labels) {
    p.real_id[idx]=lab.first ;
    idx++;
}



 if(quality > prev_quality){
    aggregate_graph(p);
  }
  else {
    cout << "DONE" << endl;
  }
    return 0;
}