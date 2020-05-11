from munkres import Munkres


def inp(x,y,D):
    # x is a dict
    res = 0
    for kv in x.items():
        res += kv[1]*y[kv[0]]
    return res
def diff(f1,f2,D):
    res = []
    for i in range(D):
        res.append(abs(f1[i]-f2[i]))
    return res

def makeSparse(feat,D):
    res = {}
    for i in range(D):
        if feat[i]:
            res[i] = feat[i]
    return res
def loss(l,lhat,N,which):
    if len(l)==0:
        if len(lhat)==0:
            return 0
        return 1.0
    if len(lhat)==0:
        if len(l)==0:
            return 0
        return 1.0
    tp = 0
    fp = 0
    fn = 0
    ll = 0
    for it in l:
        if len(lhat) and max(lhat) == it:
            fn+=1
            if which=='ZEROONE':
                ll += 1.0/N
            elif which =='SYMMETRICDIFF':
                ll += 0.5/len(l)
    for it in lhat:
        if len(l) and max(l)==it:
            fp+=1
            if which =="ZEROONE":
                ll+=1.0/N
            elif which =='SYMMETRICDIFF':
                ll+=0.5/(N-len(l))
        else:
            tp+=1
    if ((len(lhat)==0 or tp==0) and which=='FSCOREE'):
        return 1.0
    precision = (1.0*tp)/len(lhat)
    recall = (1.0*tp)/len(l)
    if which =='FSCORE':
        return 1-2*(precision*recall)/(precision+recall)
    return ll


def totalLoss(clusters,chat,N,which):
    Matrix = [[0]*len(chat) for i in range(len(clusters))]
    for i in range(len(clusters)):
        for j in range(len(chat)):
            Matrix[i][j] = loss(clusters[i],chat[j],N,which)
    m = Munkres()
    # print('matrix',Matrix)
    Indexes = m.compute(Matrix) # ??
    M  = [[-1]*len(chat) for i in range(len(clusters))]
    for item in Indexes:
        x = item[0]
        y = item[1]
        M[x][y] = 0
    l = 0
    for i in range(len(clusters)):
        for j in range(len(chat)):
            if M[i][j]==0:
                l+=loss(clusters[i],chat[j],N,which)
    size = len(clusters) if len(clusters)<len(chat) else len(chat)
    return l/size

# def compress(fol, feat,res):
#     where = 0
#     for i in range(len(fol)):
#         sum = 0
#         for j in range(fol[i]):
#             sum += feat[where]
#             where +=1
#         res[i] = sum
#
