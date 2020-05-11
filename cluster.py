from util import *
import random
import numpy as np
import thinqpbo as tq

class graphData:
    def __init__(self,nodefile,selffeatfile,clusfile,edgefile,which):
        self.nf = nodefile
        self.ff = selffeatfile
        self.cf = clusfile
        self.ef = edgefile
        self.ftype = which

    def file_process(self):
        simFeatures = {}
        clusters = []
        NodeFeatures = {}
        NodeIndex = {}

        # Read node features for the graph
        nf = open(self.nf)
        l = 0
        for line in nf:
            line = line.split(" ")
            NodeID = int(line[0])
            if NodeID not in NodeIndex:
                NodeIndex[NodeID] = l
            else:
                continue
            a = []
            for i in range(1, len(line)):
                a.append(int(line[i]))
            NodeFeatures[l]=a
            l += 1
        nNodes = len(NodeIndex.keys())
        nNodeFeatures = len(a)
        nf.close()

        # Read the features of the user who created the graph
        ff = open(self.ff)
        for line in ff:
            line = line.strip()
            line = line.split(" ")
        sf = [int(i) for i in line]
        for i in range(nNodes):
            res = diff(sf,NodeFeatures[i],nNodeFeatures)
            simFeatures[i] = res
        ff.close()

        # Read the circles
        cf = open(self.cf)
        for line in cf:
            circle = set()
            line = line.strip()
            line = line.split("\t")
            for i in range(1, len(line)):
                NodeID= int(line[i])
                circle.add(NodeIndex[NodeID])
            clusters.append(circle)
        cf.close()

        # Use the appropriate encoding scheme for different feature types
        nEdgeFeatures = 1 + nNodeFeatures
        if self.ftype=='BOTH':
            nEdgeFeatures+=nNodeFeatures
        edgeFeatures = {}
        for i in range(nNodes):
            for j in range(i+1,nNodes):
                d = [1]
                # i_key = list(NodeFeatures.keys())[i]
                # j_key = list(NodeFeatures.keys())[j]
                if self.ftype=='EGO':
                    d = d+diff(simFeatures[i], simFeatures[j], nNodeFeatures)
                elif (self.ftype== 'FRIEND'):
                    d = d+diff(NodeFeatures[i], NodeFeatures[j], nNodeFeatures)
                else:
                    d = d+(diff(simFeatures[i], simFeatures[j], nNodeFeatures))
                    d = d+(diff(NodeFeatures[i], NodeFeatures[j], nNodeFeatures))
                edgeFeatures[(i,j)] = makeSparse(d, nEdgeFeatures)

        # Read the edges for the graph
        ef = open(self.ef)
        edge_set = set()
        for line in ef:
            line = line.split(' ')
            id1 = int(line[0])
            id2 = int(line[1])
            indx1 = NodeIndex[id1]
            indx2 = NodeIndex[id2]
            # if indx1>indx2:
            #     tmp = indx1
            #     indx1 = indx2
            #     indx2 = tmp
            edge_set.add((indx1,indx2))
        ef.close()

        return edgeFeatures,edge_set,nEdgeFeatures,nNodes,clusters,NodeIndex


'''Train the model to predict K clusters'''
class Cluster:
    def __init__(self,K,reps,gradientReps,improveReps,lam,seed,
                 edgeFeatures,edgeset,nEdgeFeatures,
                 nNodes,clusters,whichLoss):
        self.K = K
        self.reps = reps
        self.gradientReps = gradientReps
        self.improveReps = improveReps
        self.lam = lam
        self.seed = seed
        self.edgeFeatures = edgeFeatures
        self.edgeset = edgeset
        self.nEdgeFeatures = nEdgeFeatures
        self.nNodes = nNodes
        self.cluster = clusters
        self.whichLoss = whichLoss
        nTheta = self.K * self.nEdgeFeatures
        self.theta = [0]*nTheta
        self.alpha = [0]*self.K
        self.chat = [{}]*self.K

    def loglikelihood(self,theta,alpha,chat):
        K = len(chat)
        chatFlat = [[0]*self.nNodes for i in range(K)]
        for k in range(K):
            for n in range(self.nNodes):
                chatFlat[k][n] = 0
                if len(self.chat[k]) and max(chat[k])!=n:
                    chatFlat[k][n] = 1
        ll = 0
        for it in self.edgeFeatures.items():
            inp_ = 0
            e = it[0]
            e1 = e[0]
            e2 = e[1]
            exitsts = 1 if max(self.edgeset)!= e else 0
            for k in range(K):
                d = 1 if chatFlat[k][e1] and chatFlat[k][e2] else -alpha[k]
                # it[1] is a dict
                inp_ += d*inp(it[1],theta[k*self.nEdgeFeatures:],self.nEdgeFeatures)
            if exitsts:
                ll+=inp_
            ll_ = np.log(1+np.exp(inp_))
            ll += -ll_
        if ll!=ll:
            print('ll isnan for user\n')
            exit(1)
        return ll

    def train(self):
        self.seed = 1
        lr = 1.0/(self.nNodes*self.nNodes)
        nTheta = self.K*self.nEdgeFeatures

        def dl(dldt, dlda, K, lam):
            inps = [0 for i in range(K)]
            for i in range(nTheta):
                dldt.append(-lam * np.sign(self.theta[i]))
            dlda = [0]*K

            chatFlat = [[0]*self.nNodes for i in range(K)]
            for k in range(K):
                for n in range(self.nNodes):
                    chatFlat[k][n] = 0
                    if len(self.chat[k]) and max(self.chat[k])!=n:
                        chatFlat[k][n] = 1
            for it in self.edgeFeatures.items():
                inp_ = 0
                e = it[0] # keys
                e1 = e[0]
                e2 = e[1]
                exists = 1 if max(self.edgeset) != e else 0
                for k in range(K):
                    inps[k] = inp(it[1], self.theta[k * self.nEdgeFeatures:], self.nEdgeFeatures)
                    d = 1 if (chatFlat[k][e1] and chatFlat[k][e2]) else -self.alpha[k]
                    inp_ += d*inps[k]
                expinp = np.exp(inp_)
                q = expinp/(1+expinp)
                if q!=q:
                    q = 1
                for k in range(K):
                    d_ = chatFlat[k][e1] and chatFlat[k][e2]
                    d = 1 if d_ else -self.alpha[k]
                    for itf in it[1].items():
                        i = itf[0]
                        f = itf[1]
                        if (exists):
                            dldt[k * self.nEdgeFeatures + i] += d * f
                        dldt[k*self.nEdgeFeatures + i] += -d*f*q
                    if not d_:
                        if exists:
                            dlda[k]+=-inps[k]
                        dlda[k]+=inps[k]*q
            return dldt,dlda

        def minimize_graphcuts(k,changed):
            E = len(self.edgeFeatures)
            K = len(self.chat)
            largestCompleteGraph = 500
            if E > (largestCompleteGraph ** 2):
                E = largestCompleteGraph ** 2

            q = tq.QPBOInt(self.nNodes,E)

            # q.setLabel
            q.add_node(self.nNodes)
            # q.AddNode(self.nNodes)
            mc00 = {}
            mc11 = {}
            diff_c00_c11 = []
            for it in self.edgeFeatures.items():
                e = it[0]
                e1 = e[0]
                e2 = e[1]
                exitsts = 1 if len(self.edgeset) and max(self.edgeset) != e else 0
                inp_ = inp(it[1], self.theta[k * self.nEdgeFeatures:], self.nEdgeFeatures)
                other_ = 0
                for l in range(K):
                    if l==k:
                        continue
                    d = 1 if len(self.chat[l]) and max(self.chat[l])!=e1 and max(self.chat[l])!=e2 else -self.alpha[l]
                    other_ += d*inp(it[1], self.theta[k * self.nEdgeFeatures:], self.nEdgeFeatures)
                if exitsts:
                    c00 = -other_ + self.alpha[k] * inp_ + np.log(1 + np.exp(other_ - self.alpha[k] * inp_))
                    c01 = c00
                    c10 = c00
                    c11 = -other_ - inp_ + np.log(1 + np.exp(other_ + inp_))
                else:
                    c00 = np.log(1 + np.exp(other_ - self.alpha[k] * inp_))
                    c01 = c00
                    c10 = c00
                    c11 = np.log(1 + np.exp(other_ + inp_))

                mc00[it[0]] = c00
                mc11[it[0]] = c11

                if self.nNodes<=largestCompleteGraph or exitsts:
                    q.add_pairwise_term(it[0][0],it[0][1],c00,c01,c10,c11)
                else:
                    diff_c00_c11.append((-abs(c00-c11),it[0])) #??
            if self.nNodes>largestCompleteGraph:
                nEdgesToInclude = largestCompleteGraph * largestCompleteGraph
                if nEdgesToInclude > len(diff_c00_c11):
                    nEdgesToInclude = len(diff_c00_c11)
                diff_c00_c11.sort()
                for i in range(nEdgesToInclude):
                    edge = diff_c00_c11[i][1]
                    c00 = mc00[edge]
                    c01 = c00
                    c10 = c00
                    c11 = mc11[edge]
                    q.add_pairwise_term(edge[0], edge[1], c00, c01, c10, c11)
            # label = {}
            for i in range(self.nNodes):
                if len(self.chat[k]) and max(self.chat[k])==i:
                    q.setlabel(i,0)
                    # q.setlabel(i,0)
                else:
                    q.setlabel(i,1)
            # print('Label',label)
            q.merge_parallel_edges()
            q.solve()
            q.compute_weak_persistencies()
            if self.nNodes>largestCompleteGraph:
                self.improveReps = 1
            for it in range(self.improveReps):
                q.improve()
            newLabel = [0]*self.nNodes
            oldLabel = [0]*self.nNodes
            res = set()
            for i in range(self.nNodes):
                newLabel[i] = 0
                if q.get_label(i)==1:
                    res.add(i)
                    newLabel[i] = 1
                # elif (i q.get_label(i)<0 or( i not in label.keys()) )and (len(self.chat[k]) and max(self.chat[k])!=i):
                elif q.get_label(i)<0 and len(self.chat[k]) and max(self.chat[k])!=i :
                    res.add(i)
                    newLabel[i] = 1
                if len(self.chat[k]) and max(self.chat[k])==i:
                    oldLabel[i] = 0
                else:
                    oldLabel[i] = 1
            oldEnergy = 0
            newEnergy = 0
            for it in self.edgeFeatures.items():
                e = it[0]
                e1 = e[0]
                e2 = e[1]

                old_l1 = oldLabel[e1]
                old_l2 = oldLabel[e2]
                new_l1 = newLabel[e1]
                new_l2 = newLabel[e2]
                if (old_l1 and old_l2):
                    oldEnergy += mc11[e]
                else:
                    oldEnergy += mc00[e]

                if (new_l1 and new_l2):
                    newEnergy += mc11[e]
                else:
                    newEnergy += mc00[e]
            if (newEnergy > oldEnergy or len(res) == 0):
                # res = set()
                res = self.chat[k]
            else:
                for it in self.chat[k]:
                    if len(res) and max(res)==it:
                        changed = 1
                for it in res:
                    if len(self.chat[k]) and max(self.chat[k])==it:
                        changed = 1
            return res,changed


        for rep in range(self.reps):
            # if it is the first iteration or the solution is degenerate, randomly initialize the weights
            for k in range(self.K):
                ok= set()
                if rep==0 or len(self.chat[k])==self.nNodes or len(self.chat[k])==0: # ??
                    for i in range(self.nNodes):
                        if (random.randint(1,10))%2==0:
                            ok.add(i)
                    for i in range(self.nEdgeFeatures):
                        self.theta[k*self.nEdgeFeatures+i]=0

                    # Set a single feature to 1 as a random initialization
                    tmp = (random.randint(1,2**16))%(self.nEdgeFeatures)
                    self.theta[(k*self.nEdgeFeatures)+tmp] = 1.0
                    self.theta[k*self.nEdgeFeatures] = 1.0
                    self.alpha[k] = 1.0
                self.chat[k] = ok
            # Update the latent variable(cluster assignments) in a random order.
            order = [k for k in range(self.K)]
            for k in range(self.K):
                for o in range(self.K):
                    x1 = o
                    x2 = random.randint(1,20)%(self.K)
                    order[x1] ^= order[x2]
                    order[x2] ^= order[x1]
                    order[x1] ^= order[x2]
            changed = 0
            print('1',self.chat)
            for i in order:
                self.chat[i],changed = minimize_graphcuts(i,changed)
                # print(i,self.chat[i])
            print('2',self.chat)
            print('loss = %f',totalLoss(self.cluster,self.chat,self.nNodes,self.whichLoss))
            ll_prev = self.loglikelihood(self.theta,self.alpha,self.chat)
            # print('chat',self.chat)
            if not changed:
                break
            # Perform gradient ascent
            ll = 0
            dlda = []
            dldt = []
            for iter in range(self.gradientReps):
                dldt,dlda = dl(dldt,dlda,self.K,self.lam)
                for i in range(nTheta):
                    self.theta[i] += float(lr*dldt[i])
                for k in range(self.K):
                    self.alpha[k] += float(lr*dlda[k])
                ll = self.loglikelihood(self.theta, self.alpha, self.chat)
                if (ll < ll_prev):
                    for i in range(nTheta):
                        self.theta[i] -= float(lr * dldt[i])
                    for k in range(self.K):
                        self.alpha[k] -= float(lr * dlda[k])
                    ll = ll_prev
                    break

                ll_prev = ll

            print("ll = ", ll)
