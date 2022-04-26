"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt

n=24 # 节点数
inf=100000
graph = [[(lambda x: 0 if x[0] == x[1] else inf)([i, j]) for j in range(n)] for i in range(n)] #两点间距离
parents = [[i] * n for i in range(n)]


vnfs = np.random.randint(5,size=(n, 8,3)) #Type, resource space of eight vnfs of each node
vnfs1=np.zeros((n, 8), dtype=int) #Types of eight vnfs per node
for u in range(n):
    for v in range(8):
        vnfs[u][v][1]=115
        vnfs[u][v][2]=np.random.randint(8, 13)
        vnfs1[u][v]=vnfs[u][v][0]

'''
class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(TSPDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        self.dataset = torch.rand((num_samples, 4, size))
        self.dynamic = torch.zeros(num_samples, 1, size)
        self.num_nodes = size
        self.size = num_samples


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])
'''

class TSPDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(TSPDataset, self).__init__()


        gg=g0.copy()
        for u, v, c in data2:
            gg[v][c] = 2000
        gg=torch.tensor(gg)
        vv=vnfs1.copy()
        vv = torch.tensor(vv)
        self.dataset = torch.cat((gg,vv),dim=1)
        self.dataset = self.dataset.permute(1, 0)
        self.dataset = self.dataset.expand(num_samples,-1,-1)

        self.dynamic = torch.zeros(num_samples, 1, n)
        self.num_nodes = n
        self.size = num_samples


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [])

class chainDataset(Dataset):

    def __init__(self, size=50, num_samples=1e6, seed=None):
        super(chainDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(123456789)

        np.random.seed(seed)
        torch.manual_seed(seed)
        l=np.random.randint(195, 260)
        p = np.array([0.25, 0.75])
        chain = [[[np.random.randint(0, 5), np.random.randint(10, 20) * (0 if i==l-1 else np.random.choice([0, 1], p=p.ravel())),np.random.randint(10, 40)] for i in range(l)] for j in range(num_samples)]
        self.dataset = torch.tensor(chain)

        self.dynamic = torch.zeros(num_samples, 1, l)
        self.num_nodes = n
        self.size = num_samples


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx])

def update_mask(mask, dynamic, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

'''
def reward(static, tour_indices, w1=1, w2=0):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand_as(static)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Make a full tour by returning to the start
    y = torch.cat((tour, tour[:, :1]), dim=1)
    # first 2 is xy coordinate, third column is another obj
    y_dis = y[:, :, :2]
    y_dis2 = y[:, :, 2:]

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y_dis[:, :-1] - y_dis[:, 1:], 2), dim=2))
    obj1 = tour_len.sum(1).detach()

    tour_len2 = torch.sqrt(torch.sum(torch.pow(y_dis2[:, :-1] - y_dis2[:, 1:], 2), dim=2))
    obj2 = tour_len2.sum(1).detach()

    obj = w1*obj1 + w2*obj2
    return obj, obj1, obj2
'''

def reward(chain,tour_indices,w1=1,w2=0):


    obj =[]
    chanin_index =0
    for tour in tour_indices:
        objtem=0
        obj_r=0
        g0tem = g0.copy()
        g01={}
        g02={}
        g1tem = g1.copy()
        g2tem = g2.copy()
        vnftem = vnfs.copy()  # temp vnfs
        global parents

        parents = [[i] * n for i in range(n)]
        nodestate=np.zeros(n, dtype=int)
        vnfstate=np.zeros((n,8), dtype=int)
        for i in range(len(tour)):

            flag = 0
            for j in range(len(vnftem[tour[i]])):
                if (vnftem[tour[i]][j][0] == chain[chanin_index][i][0]) & (vnftem[tour[i]][j][1] >= chain[chanin_index][i][2]):
                    flag = 1
                    vnftem[tour[i]][j][1] -= chain[chanin_index][i][2].item()
                    if nodestate[tour[i]] ==0:
                        nodestate[tour[i]] = 1
                    if vnfstate[tour[i]][j]==0:
                        vnfstate[tour[i]][j] = 1
                    break
            if flag == 0:
                objtem += chain[chanin_index][i][2].item()
            flag = 0

            if i==0:
                continue
            elif chain[chanin_index][i-1][1]==0:
                continue
            elif tour[i-1].item()==tour[i].item():
                continue
            else:
                floyd(0,g0tem,g01,g02,g1tem,g2tem,chain[chanin_index][i-1][1].item())
                path=print_path(tour[i-1].item(),tour[i].item())
                if len(path) != 0:
                    for p in range(len(path)):
                        nodestate[path[p]]=1
                        if p !=0:
                            g0tem[path[p-1]][path[p]]-=chain[chanin_index][i-1][1]
                else:
                    floyd(1, g0tem,g01,g02, g1tem, g2tem, chain[chanin_index][i-1][1].item())
                    path =print_path(tour[i - 1].item(), tour[i].item())
                    if len(path) != 0:
                        tpath=[]
                        for p in range(len(path)):
                            nodestate[path[p]] = 1
                            if p != 0:
                                if [path[p-1],path[p]] in g1tem:
                                    if (len(tpath)==0):
                                        tpath.append([path[p - 1],path[p]])
                                    elif (tpath[-1][-1]==path[p - 1]):
                                        tpath[-1][-1] = path[p]
                                    elif (tpath[-1][-1] != path[p - 1]):
                                        tpath.append([path[p - 1],path[p]])
                                    g1tem.remove([path[p-1],path[p]])
                                elif (path[p-1],path[p]) in g01:
                                    g01[path[p-1],path[p]]-=chain[chanin_index][i-1][1]
                                else:
                                    g0tem[path[p-1]][path[p]]-=chain[chanin_index][i-1][1]
                        for p in tpath:
                            g01[p[0],p[1]]=2000-chain[chanin_index][i-1][1]
                    else:
                        floyd(2, g0tem,g01,g02, g1tem, g2tem, chain[chanin_index][i-1][1].item())
                        path =print_path(tour[i - 1].item(), tour[i].item())
                        if len(path) != 0:
                            tpath = []
                            for p in range(len(path)):
                                nodestate[path[p]] = 1
                                if p != 0:
                                    if [path[p - 1], path[p]] in g2tem:
                                        if (len(tpath) == 0):
                                            tpath.append([path[p - 1], path[p]])
                                        elif (tpath[-1][-1] == path[p - 1]):
                                            tpath[-1][-1] = path[p]
                                        elif (tpath[-1][-1] != path[p - 1]):
                                            tpath.append([path[p - 1], path[p]])
                                        g2tem.remove([path[p - 1], path[p]])
                                    elif (path[p-1],path[p]) in g02:
                                        g02[path[p - 1], path[p]] -= chain[chanin_index][i - 1][1]
                                    else:
                                        g0tem[path[p - 1]][path[p]] -= chain[chanin_index][i-1][1]
                            for p in tpath:
                                g02[p[0],p[1]]=2000-chain[chanin_index][i-1][1]
                        else:
                            objtem += chain[chanin_index][i-1][1].item()
        obj_r=revenue(g0tem,g01,g02,vnftem,nodestate,vnfstate)
        obj.append(objtem*100000+obj_r)
        chanin_index += 1
        #print(obj)
    obj = torch.tensor(obj).detach()
    obj = obj.to(torch.float32)
    obj1 =obj
    obj2 =torch.zeros(len(obj)).detach()
    return obj,obj1,obj2

def getRos(n,vnftem):
    res=0
    up=0
    down=0
    for i in range(8):
        up += (115-vnftem[n][i][1])
        down += 115
    res = up/down
    return res

def geto(u,v,vnf):
    res =1000*100
    tp=0
    for i in range(8):
        tp += (115-vnf[u][i][1])
        tp+=(115-vnf[v][i][1])
    return res/(tp+1)

# the objective vlaue（accord to utility in this paper） after sfc embedded
def revenue(g0,g01,g02,vnftem,node,vnf):
    result=0;un=0;zn=0;ze=0;zo=0
    for i in range(n): # U,zn
        if node[i]==1:
            zn+=50
            rs = getRos(i,vnftem)
            for j in range(8):
                if vnf[i][j]==1:
                    zn+=200
                    zn+=rs*(115-vnftem[i][j][1])
                    un+=(115-vnftem[i][j][1])*vnftem[i][j][2]
    for u,v,w in data1:
        if g0[v][w]<400:
            ze+=10
            ze+=((400-g0[v][w])*(400-g0[v][w]))/400
        if g0[w][v]<400:
            ze+=10
            ze+=((400-g0[w][v])*(400-g0[w][v]))/400
    for u,v in g01:
        zo +=geto(u,v,vnftem)
    for u,v in g02:
        zo +=geto(u,v,vnftem)
    result = zn+ze+zo-un
    return result

def graphup(w,g0,g01,g02,g1,g2,b):
    g0temp=g0.copy()
    g1temp = g1.copy()
    g2temp = g2.copy()
    g01temp=g01.copy()
    g02temp=g02.copy()
    if w==1:
        for u,v in g1temp:
            g0temp[u][v]=2000
        for u,v in g01temp:
            g0temp[u][v] = g01temp[u,v]
    if w==2:
        for u,v in g2temp:
            g0temp[u][v]=2000
        for u,v,w in g02temp:
            g0temp[u][v] = g02temp[u,v]
    for i in range(n):
        for j in range(n):
            if i==j:
                graph[i][j]=0
            elif (g0temp[i][j] == inf) | (g0temp[i][j] < b):
                graph[i][j] = inf
            elif g0temp[i][j] >= b:
                graph[i][j] = 1


def floyd(w,g0temp,g01,g02,g1temp,g2temp,b):
    graphup(w,g0temp,g01,g02,g1temp,g2temp,b)
    for i in range(n):
        for j in range(n):
            if graph[i][j] != inf:
                parents[i][j]=j
            else:
                parents[i][j] = -1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if (k != i)&(i != j):
                    if graph[i][k] + graph[k][j] < graph[i][j]:
                        graph[i][j] = graph[i][k] + graph[k][j]
                        parents[i][j] = parents[i][k]  # 更新父结点


# 打印路径
def print_path(i, j):
    if graph[i][j] >=inf:
        return []
    path=[]
    end =parents[i][j]
    path.append(i)
    while(end!=j):
        path.append(end)
        end = parents[end][j]
        if(len(path)>=20):
            print(path)
    path.append(j)
    return path


# Electrical link data
data1=[
[0, 0, 1],
[1,0, 5],
[2, 1, 2],
[3, 1, 5],
[4, 2, 6],
[5, 2, 4],
[6, 2, 3],
[7, 3, 4],
[8, 3, 6],
[9, 4, 7],
[10, 5, 6],
[11, 5, 8],
[12, 5, 10],
[13, 6, 7],
[14, 6, 8],
[15, 7, 9],
[16, 8, 9],
[17, 8, 10],
[18, 8, 11],
[19, 9, 12],
[20, 9, 13],
[21, 10, 11],
[22, 10, 14],
[23, 10, 18],
[24, 11, 15],
[25, 11, 12],
[26, 12, 13],
[27, 12, 16],
[28, 13, 17],
[29, 14, 15],
[30, 14, 19],
[31, 15, 16],
[32, 15, 20],
[33, 15, 21],
[34, 16, 21],
[35, 16, 22],
[36, 16, 17],
[37, 17, 23],
[38, 18, 19],
[39, 19, 20],
[40, 20, 21],
[41, 21, 22],
[42, 22, 23]
]
# Optical link data
data2=[
[0, 14, 5],
[1, 4, 20],
[2, 13, 17],
[3, 12, 10],
[4, 15, 2],
[5, 9, 3],
[6, 10, 0],
[7, 2, 18],
[8, 16, 3],
[9, 4, 0],
[10, 5, 22],
[11, 13, 7],
[12, 3, 12],
[13, 7, 11]
]

g0=np.zeros((n, n), dtype=int) # 电链路
#data1=[]
#data2=[]
for u, v, c in data1:
    g0[v][c] = 400
    g0[c][v]=400

g1=[] #存储光链路
g2=[]
for u, v, c in data2:
    g1.append([v,c])
    g2.append([v,c])
