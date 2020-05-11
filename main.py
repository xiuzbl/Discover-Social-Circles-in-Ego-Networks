from cluster import graphData, Cluster
from util import *


def experiment(gd,K,lam,reps,gradientReps,improveReps):
    bestll = 0
    seed = 42
    edgeFeatures,edge_set,nEdgeFeatures,nNodes,clusters,NodeIndex = gd.file_process()
    print(NodeIndex,nEdgeFeatures,nNodes,clusters)
    C = Cluster(K,reps,gradientReps,improveReps,lam,seed,edgeFeatures,
                edge_set,nEdgeFeatures,nNodes,clusters,whichLoss='SYMMETRICDIFF')
    print('cluster',C.cluster)
    nseeds = 1 # Number of random restarts
    for seed in range(nseeds):
        seed+=1
        C.train()
        ll = C.loglikelihood(C.theta,C.alpha,C.chat)
        print(C.chat)
        if ll>bestll or bestll==0:
            bestll = ll
            bestClusters = C.chat
            bestTheta = C.theta
            bestAlpha = C.alpha

    file = open('result.txt','w')
    print('ll = ',bestll,file=file)
    print('loss_zeroone = ', totalLoss(clusters, bestClusters, nNodes, 'ZEROONE'), file=file)
    print("loss_symmetric = ", totalLoss(clusters, bestClusters, nNodes, 'SYMMETRICDIFF'), file=file)
    print("fscore = ", 1 - totalLoss(clusters, bestClusters, nNodes, 'FSCORE'), file=file)
    print('Clusters:\n',bestClusters,file=file)
    print('Theta:\n', bestTheta, file=file)
    print('Alpha:\n', bestAlpha, file=file)

# if __name__=='main':

ID = 'facebook/'+'698'
nodefile = ID+'.feat'
selffeatfile = ID+'.egofeat'
clusfile = ID+'.circles'
edgefile = ID+'.edges'

gd = graphData(nodefile, selffeatfile, clusfile, edgefile, 'FRIEND')
experiment(gd,3,1,25,50,5)
