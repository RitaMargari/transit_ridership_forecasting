import numpy as np
import networkx as nx
import subprocess, os
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
#import geopandas as gpd
import math
#from community import community_louvain
import pickle

import urllib.request as urllib
import urllib.request as urllib
import io
import zipfile
import time
import warnings

import pycombo

##V 6.9.21 #added hierarchical spectral partition

def spectralBiPartition(A): #bi-partitmodulaion of the network based on the second eigenvector of Laplacian
    if A.shape[0]>1:
        d = np.sum(A, axis=1)
        L = np.diag(d) - A
        w, v = np.linalg.eig(L)
        c = v[np.argsort(w)[-2],:] > 0
    else:
        c = np.zeros(1)
    return c #array of zeros-ones according to partition

def spectralHierPartition(A, maxdep = 100): #hierarchical spectral partition of the network
    if A.shape[0] > 1:
        c = spectralBiPartition(A)
        A1 = A[c==0, :][:, c==0]
        A2 = A[c==1, :][:, c==1]
        if maxdep > 1:
            h1 = spectralHierPartition(A1, maxdep = maxdep-1)
            h2 = spectralHierPartition(A2, maxdep = maxdep-1)
            h = np.zeros((max(h1.shape[0], h2.shape[0]) + 1, len(c)))
            h[1:(h1.shape[0] + 1), c==0] = h1
            h[1:(h2.shape[0] + 1), c==1] = h2
        else:
            h = np.zeros((1,len(c)))
        h[0, :] = c
    else:
        h = np.zeros((1,1))
    return h


##V 1.09.21 #added matlab network loader and TimeTracker

def pickleDump(fname, vars ,protocol=3):
    if len(fname) > 0:
        with open(fname, 'wb') as f:
            pickle.dump(vars, f, protocol=protocol)

def pickleLoad(fname):
    if len(fname) > 0:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    else:
        return []

class TimeTracker(object): #time tracking
    startTime={}
    totalTime={}
    def startActivity(self,actID):
        self.startTime[actID] = time.time()
        if actID not in self.totalTime.keys():
            self.totalTime[actID] = 0
    def stopActivity(self,actID):
        self.totalTime[actID]+=time.time()-self.startTime[actID]
    def reportActivity(self,actID):
        print('Elapsed time for activity {}={}'.format(actID, self.totalTime[actID]))
    def reportTotal(self):
        for actID in self.totalTime.keys():
            self.reportActivity(actID)

def accumarray2d(xy, vals, dims = (0,0)): #accumarray for 2d coordinates (edges), values to aggregate vals and dimensions dims
    xy = np.array(xy)
    dims = (max(max(xy[:,0]) + 1, dims[0]), max(max(xy[:,1]) + 1, dims[1]))
    xy_ = dims[1] * xy[:,0] + xy[:,1]
    return np.bincount(xy_, weights=vals, minlength=dims[0]*dims[1]).reshape(dims)


#visualize network partitioning given node locations pos
def modularity_old(G,partition):
    #does not work correctly for negative weights
    #compute network modularity according to the given partitioning
    nodes=list(G.nodes())
    #compute node weights and total network weight
    if G.is_directed():
        w1=G.out_degree(weight='weight')
        w2=G.in_degree(weight='weight')
        T=1.0*sum([e[2]['weight'] for e in G.edges(data=True)])
    else:
        w1=G.degree(weight='weight')
        w2=G.degree(weight='weight')
        T=1.0*sum([(1+(e[0]!=e[1]))*e[2]['weight'] for e in G.edges(data=True)])
    M=0 #start accumulating modularity score
    for a in nodes:
        for b in nodes:
            #if (G.is_directed())|(b>=a):
                if partition[a]==partition[b]: #if nodes belong to the same community
                    #get edge weight
                    if G.has_edge(a,b):
                        e=G[a][b]['weight']
                    else:
                        e=0
                    M+=e/T-w1[a]*w2[b]/(T**2) #add modularity score for the considered edge
    return M

def modularity(G, partition, correctLoops = False): #modularity of the networkx graph given partition dictionary
    Q = getModularityMatrix(G, correctLoops = correctLoops)
    C = np.array([partition[n] for n in G.nodes()]) #could there be an indexing mismatch between Q and C
    return (Q * (C.reshape(-1,1) == C.reshape(1,-1))).sum()


def matrixToModularity(A):
    w1 = np.sum(A, axis=1)
    w2 = np.sum(A, axis=0)
    return (A - np.matmul(w1.reshape((-1,1)),w2.reshape((1,-1))))/np.sum(A)

def modularityMatrix(G):
    #compute network modularity according to the given partitioning and returns it as a graph
    nodes=list(G.nodes())
    #compute node weights and total network weight
    if G.is_directed():
        w1=G.out_degree(weight='weight')
        w2=G.in_degree(weight='weight')
        T=1.0*sum([e[2]['weight'] for e in G.edges(data=True)])
    else:
        w1=G.degree(weight='weight')
        w2=G.degree(weight='weight')
        T=1.0*sum([(1+(e[0]!=e[1]))*e[2]['weight'] for e in G.edges(data=True)])
    M=G.copy() #start accumulating modularity score
    for a in nodes:
        for b in nodes:
                if G.has_edge(a,b):
                    M[a][b]['weight']=G[a][b]['weight']/T-w1[a]*w2[b]/(T**2) #add modularity score for the considered edge
                else:
                    M.add_edge(a,b,weight=-w1[a]*w2[b]/(T**2)) #add modularity score for the considered edge
    return M

def getModularityMatrix(G, symmetrize = False, correctLoops = False):  # build mobilarity matrix and return as numpy
    A = np.array(nx.adjacency_matrix(G).todense(), dtype = float)
    if correctLoops and not isinstance(G,nx.DiGraph):
        A += np.diag(np.diag(A))
    wout = A.sum(axis=1)
    win = A.sum(axis=0)
    T = wout.sum()
    Q = A / T - np.matmul(wout.reshape(-1, 1), win.reshape(1, -1)) / (T ** 2)
    if symmetrize:
        Q = (Q + Q.transpose()) / 2
    return Q

def visualizePartition(G,partition,pos=None):
    N=len(G.nodes())
    #s=4+4*int(log10(N))
    plt.figure(figsize=(12,12))
    PN=max(partition.values())
    my_cmap = plt.cm.hsv(np.linspace(0,1,PN+1)) #create a colormap for a given number of communities
    c=[]
    for n in G.nodes():
        c.append(1.0*partition[n]/PN)
    nx.draw(G,pos=pos,with_labels=False,arrows=True,node_size=300,node_color=c,width=1,edge_color='black')
    plt.show()

def make_weighted_(G):
    WG=G
    for e in WG.edges(): #check if already has weight
        WG[e[0]][e[1]]['weight']=1
    return WG

def make_weighted(G):
    WG=G
    for e in WG.edges(): #check if already has weight
        if 'weight' not in WG[e[0]][e[1]].keys():
            if 'value' in WG[e[0]][e[1]].keys():
                WG[e[0]][e[1]]['weight'] = WG[e[0]][e[1]]['value']
            else:
                WG[e[0]][e[1]]['weight'] = 1
    return WG

def totalWeight(G,f=lambda x: x):
    return sum([f(d['weight']) for _, _, d in G.edges(data=True)])

def partitionGraph(G,P):
    GP=[None]*P.shape[1]
    for p in range(P.shape[1]):
        nzind=np.where(P[:,p]>1e-6)[0]
        P_={list(G.nodes)[i]:P[i,p] for i in nzind}
        GP[p]=G.subgraph(P_.keys())
        for e in GP[p].edges():  # check if already has weight
            GP[p][e[0]][e[1]]['weight'] *= P_[e[0]]*P_[e[1]]
    return GP

def partitionSeries(G, ItN = 10, method = 'combo', verbose = 0, loopCorrection = True):
    if ItN>0:
        if loopCorrection and not isinstance(G,nx.DiGraph):
            G = G.copy()
            for a in G.nodes():
                if G.has_edge(a, a):
                    G[a][a]['weight'] /= 2.0
        parts = []
        mod = np.zeros((ItN, 2))
        for i in range(ItN):
            if method == 'combo':
                part = getComboPartition(G, 100)
            else:
                part = community_louvain.best_partition(G)
            mod[i,0] = modularity(G, part, correctLoops=loopCorrection)
            mod[i,1] = max(list(part.values()))+1
            parts += [part]
            if verbose>1:
                print(method + ' modularity={}; {} comm'.format(modularity(G, part, correctLoops=loopCorrection), max(list(part.values()))))
        i = np.argmax(mod[:,0])
        if verbose > 0:
            print('Best ' + method + ' modularity={}; {} comm'.format(mod[i,0],mod[i,1]))
        res = {'best': mod[i,0], 'best1': mod[0,0], 'best5': max(mod[:5,0]), 'best10': max(mod[:10,0]), 'best20': max(mod[:20,0]), 'min': mod[:,0].min(), 'mean': mod[:,0].mean(), 'MC': int(mod[i,1]), 'partition': parts[i]}
    else:
        res = {}
    return res #max, min, mean mod and best community number


def getComboPartition(G, maxcom=0, suppressCppOutput = False):  # !!!redo with faster matrix handling, interal timer and empty node control
    # save network in net format
    # workfolder=os.path.dirname(os.path.realpath(__file__))+"/"
    workfolder = '/Users/stanislav/Desktop/MY_WORKSPACE/MIT5/NA/COMMON/'
    # create a dictionary transforming nodes to unique numbers
    nodes = list(G.nodes())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nx.write_pajek(G, workfolder + 'combo/temp.net')

    # run combo
    command = workfolder + 'combo/comboCPP ' + workfolder + 'combo/temp.net ' + (
        str(maxcom) if maxcom > 0 else 'INF') + ' 1 comm_comboC++ 0 3'

    if suppressCppOutput:
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(command, shell=True)

    # read resulting partition
    f = open(workfolder + 'combo/temp_comm_comboC++.txt', 'r')
    i = 0
    partition = {}
    for line in f:
        partition[nodes[i]] = int(line)
        i += 1
    return partition

def getComboPartition_old(G,maxcom=0): #!!!redo with faster matrix handling, interal timer and empty node control
    #save network in net format
    nodes={}
    nodenum={}
    i=0
    workfolder='/Users/stanislav/Desktop/MY_WORKSPACE/MIT5/NA/COMMON/'
    #create a dictionary transforming nodes to unique numbers
    for n in list(G.nodes()):
        nodenum[n]=i
        nodes[i]=n
        i+=1

    f = open(workfolder+'combo/temp.net', 'w')
    f.write('*Arcs\n')
    for e in G.edges(data=True):
        f.write('{0} {1} {2}\n'.format(nodenum[e[0]],nodenum[e[1]],e[2]['weight']))
    f.close()
    #run combo
    command=workfolder+'combo/comboCPP '+workfolder+'combo/temp.net '+(str(maxcom) if maxcom > 0 else 'INF')+' 1 comm_comboC++ 0 3'
    if maxcom<np.inf:
        command=command+' {0}'.format(maxcom)
    os.system(command)
    #read resulting partition
    f = open(workfolder+'combo/temp_comm_comboC++.txt', 'r')
    i=0
    partition={}
    for line in f:
        partition[nodes[i]]=int(line)
        i+=1
    return partition

def getNewComboPartition(G, maxcom=-1, suppressCppOutput = False):
    #https://pypi.org/project/pycombo/
    partition, modularity = pycombo.execute(G, return_modularity=True, max_communities = maxcom, random_seed=42)
    return modularity, partition

def getNewComboSeries(G,maxcom,tries=5,verbose=0):
    part = [None]*tries
    M = np.zeros(tries)
    for i in range(tries):
        M[i], part[i]=getNewComboPartition(G,maxcom)
        #M[i] = modularity(G,part[i])
        if verbose>0:
            print('Combo try {},mod={:.6f}'.format(i+1,M[i]))
    return part[np.argmax(M)]

def getComboSeries(G,maxcom,tries=5,verbose=0):
    part=[None]*tries
    M=np.zeros(tries)
    for i in range(tries):
        part[i]=getComboPartition(G,maxcom)
        M[i]=modularity(G,part[i])
        if verbose>0:
            print('Combo try {},mod={:.6f}'.format(i+1,M[i]))
    return part[np.argmax(M)]

def scrapeZIP(netname="",ext='gml',path="http://www-personal.umich.edu/~mejn/netdata/"):
    url = path+netname+".zip"
    sock = urllib.urlopen(url)  # open URL
    s = io.BytesIO(sock.read())  # read into BytesIO "file"
    sock.close()
    zf = zipfile.ZipFile(s)  # zipfile object
    txt = zf.read(netname+'.'+ext).decode()  # read data
    return txt.split('\n')[0:]

def readFile(netname="",ext='net', path="/Users/stanislav/Desktop/NYU/NYURESEARCH/STRATEGIC_RESEARCH/ModularityMaximum/SampleNetworks/"):
    file = open(path+netname+'.'+ext, "r")
    #txt=file.readlines()
    txt=file.read().splitlines()
    file.close()
    return txt

def parseNet(txt,spec,delimiter=None):
    if 's' in spec:
        create_using = nx.Graph
    else:
        create_using = nx.DiGraph
    if 'w' in spec:
        G = nx.read_weighted_edgelist(txt, delimiter=delimiter, create_using=create_using) #what about directionality?
    else:
        G = nx.parse_edgelist(txt, data=False, create_using=create_using)
        G=make_weighted(G)
    return(G)

def loadNetwork(netname):

    multigraph=0
    if netname[1]=='nx':
        if netname[0]=='karate':
            G = make_weighted(nx.karate_club_graph())
        elif netname[0]=='gnm':
            G = make_weighted(nx.gnm_random_graph(50, 20))
        return G
    act=False
    if netname[1] == "UM":
        act=True
        collection_path = "http://www-personal.umich.edu/~mejn/netdata/"
        if len(netname)>2:
            if netname[2]=='mult':
                multigraph=1
        ext='gml'
    elif netname[1] == "AR":
        act=True
        collection_path = 'http://deim.urv.cat/~alexandre.arenas/data/xarxes/'
        ext='net'
    if act:
        txt=scrapeZIP(netname[0],collection_path,ext)
        if ext=='gml':
            gml = txt[1:]
            if multigraph:
                gml = gml[:2] + ['multigraph 1'] + gml[2:]  # fix for multigraphs
            G = nx.parse_gml(gml)
            return G
        elif ext=='net':
            pass
    if netname[1] == 'Local':
        path = "/Users/stanislav/Desktop/NYU/NYURESEARCH/STRATEGIC_RESEARCH/ModularityMaximum/SampleNetworks/"
        txt=readFile(netname[0],ext=netname[2],path=path)
        if netname[2]=='net':
            G=parseNet(txt[3:],netname[3])
        if netname[2]=='csv':
            G = parseNet(txt, netname[3])
        if netname[2]=='gml':
            G = make_weighted(nx.read_gml(path+netname[0]+'.gml', label = 'id'))
        return G
    if netname[1] == 'Local2':
        path = '/Users/stanislav/Desktop/NYU/ADS2019/session2019_11_DL/CommunityDetection/'
        txt = readFile(netname[0], ext=netname[2], path=path)
        return [parseNet(txt[1:], netname[3],delimiter=','),readGeo(netname[4],path)]

def readGeo(fname='',path=''):
    return gpd.read_file(path + fname +'.shp')

def visualizePartitionShape(gdf, Y, idfield, offset=0, plttitle = 'partition'): #take geopandas dataframe gdf with zip code shapefiles and the dictionary mapping zip codes to clusters
    #visualize shapes using communities for picking colors
    colors=['green','blue','red','yellow','magenta']
    f, ax = plt.subplots(1, figsize=(12, 12))
    for c in range(max(Y.values())+1): #for each cluster
        if idfield=='index':
            ID=gdf.index
        else:
            ID=gdf[idfield]
        gdf.loc[(ID+offset).map(Y)==c].plot(axes=ax, color=colors[c]) #visualize zip codes which belong to it using cluster color
    plt.title(plttitle)
    plt.show()

def graphLabelsToNumeric(G,offset=0):
    #change string labeling of nodes to numeric;
    #offset shifts the numbers (e.g. offset=-1 will shift 1,2,3,... numbering to 0,1,2,...)
    mapping={n: int(n)+offset for n in G.nodes()}
    return nx.relabel_nodes(G,mapping)

def randomSample(G,s):
    ind = np.random.choice(G.nodes, replace=False, size=s)
    return G.subgraph(ind)

### fuzzy subpartition aggregation
def stackSubpartitions(P1,P2): #stack subparts from list P2 of tuples (node subindex, partition) to partition P1 into single subparition (assumed numpy arrays)
    try:
        s=[p.shape[1] for (_,p) in P2] #subpart dimensions
    except:
        1==1
    M=len(P2) #need to be consistent with P1.shape[1]
    P=np.zeros((P1.shape[0],sum(s)))
    si=0
    subpartP=np.zeros((sum(s),M)) #partition for the aggregated matrix
    for i in range(M):
        P[P2[i][0],si:(si+s[i])]=P2[i][1]*P1[P2[i][0],i].reshape(-1,1)
        subpartP[si:(si + s[i]), i] = 1
        si+=s[i]
    return P,subpartP

def sliceNetMatrix(A,P): #slice network matrix by fuzzy partition
    #### !!! embed modularity transform here - debug
    AS=[]
    for i in range(P.shape[1]):
        p_=P[:,i].reshape(-1,1)
        A_=A*np.matmul(p_,p_.transpose())
        ind=np.where(np.any(A_!=0,axis=1) | np.any(A_!=0,axis=0))[0]
        AS+=[[ind,matrixToModularity(A_[np.ix_(ind,ind)])]]
    return AS

def aggregateNetMatrix(A,P): #aggegate network matrix by fuzzy partition
    return np.matmul(np.matmul(P.transpose(),A),P)

def disaggregatePartition(Pagg,Pinit): #disaggrgate aggregated partition Pagg to the initial scale by parition used for aggregation Pinit
    return np.matmul(Pinit,Pagg)

def createNework(A, Nodes): #create a network from the adjacency matrix A and Node list to name the nodes
    G = nx.from_numpy_array(A)
    return nx.relabel_nodes(G,{i:Nodes[i] for i in range(len(G))})

def comparePart(part1,part2): #compare two partitions
    n1 = set(part1.keys())
    n2 = set(part2.keys())
    n0 = n1.intersection(n2)
    c1 = np.array([part1[n] for n in n0])
    c2 = np.array([part2[n] for n in n0])
    C1 = c1.reshape(-1,1) == c1.reshape(1,-1)
    C2 = c2.reshape(-1,1) == c2.reshape(1,-1)
    return (C1 == C2).mean()

import scipy.io

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def loadNetworkMat(filename, path = '/Users/stanislav/Desktop/NYU/NYURESEARCH/STRATEGIC_RESEARCH/ModularityMaximum/SampleNetworks/ProcessedMat/'):
    A = scipy.io.loadmat(path + filename)
    if check_symmetric(A['net']):
        G = nx.from_numpy_matrix(A['net'])
    else:
        G = nx.from_numpy_matrix(A['net'], create_using=nx.DiGraph)
    return G

def adjacency_matrix(G):
    return np.array(nx.adjacency_matrix(G).todense())

def saveNetworkMat(filename, G, path = '/Users/stanislav/Desktop/NYU/NYURESEARCH/STRATEGIC_RESEARCH/ModularityMaximum/SampleNetworks/ProcessedMat/'):
    A = adjacency_matrix(G)
    scipy.io.savemat(path + filename+'.mat', {'net' : A})

def symmetrize(G, style = 0): #transform a graph to a symmetric and positive-corrected modularity matrix
    # (enables applying community detection methods using symmetric nets; as modularity of this one is identical to the original symmetrized modularity
    if isinstance(G,nx.DiGraph) & (style >= 0):
        if style == 0: #simply symmetrize - does not respect modularity scores
            A = adjacency_matrix(G)
            A = (A + A.transpose()) / 2.0
        elif style == 1: #simmetrize respecting the modularity scores (some edges may become negative)
            A = getModularityMatrix(G, symmetrize=True)
            n = len(G)
            A += 1.0 / (n ** 2)
        elif style == 2:
            Q1 = getModularityMatrix(G, symmetrize=True)
            qm = -Q1.min()
            n = len(G)
            Q1 += qm + np.eye(n) * (1.0 / n - n * qm)
            T = Q1.min() * (len(G) ** 2)
            A = (Q1 * np.abs(T) - Q1.min())
        return nx.from_numpy_matrix(A)
    else:
        return G

def NMI(part1,part2): #NMI between two partition dictionaries
    nodes = set(part1.keys()).intersection(set(part2.keys()))
    p1 = [part1[p] for p in nodes]
    p2 = [part2[p] for p in nodes]
    return normalized_mutual_info_score(p1, p2)

networkNames={'karate':["karate",'nx'], #34
               'football':["AmericanCollegeFootball/football",'Local','csv','s'],
               'jazz':['JazzMusicians/jazz','Local','net','sw'],
               'celegan':['NeuralNetworkCElegans/celegansneural','Local','csv','sw'],
               'email':['EmailNetwork/email','Local','csv','sw'],
               'dolphins':['DolphinSocialNetwork/dolphins','Local','csv','s'],
               'lesmis':['LesMiserables/lesmis','Local','csv','sw'],
               'condmat2005':['CondensedMatterCollaboration2005/cond-mat-2005','Local','csv','sw'],
               'astro':["AstrophysicsCollaborations/astro-ph",'Local','csv','sw'],
               'polblogs':["PoliticalBlogs/polblogs",'Local','gml',''], #duplicated edges, mismatch in network size
               'USair':["USAirports/USairport_2010",'Local','csv','s'],
               'polbooks':["PoliticalBooks/polbooks",'Local','gml','s'],
               'USmig':["RNetData/USstates_migration",'Local2','csv','w','USA_adm/USA_states_reduced'],
               'NYtaxi':["RNetData/NYCnet",'Local2','csv','w','taxi_zones/taxi_zones'],
               'celeganNeural':['NeuralNetworkCElegans/celegansneural','Local','gml'], #297 weighted and directed, but weigts are given as "values"
               'netSci':['CoauthorsInNetworkScience/netscience','Local','gml'], #1589, weighted
               'powerGrid':['USPowerGrid/power','Local','gml'],
               'copperfield':['adjnoun/adjnoun','Local','gml']
              }