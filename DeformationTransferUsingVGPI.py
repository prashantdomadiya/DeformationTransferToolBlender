

bl_info = {
    "name": "Deformation Transfer",
    "author": "Prashant Domadiya",
    "version": (1, 0),
    "blender": (2, 67, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Transfer Deformation From Sorce Temporal Sequence to Target Temporal Sequence",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

import bpy
import numpy as np
import os
from scipy import sparse as sp
from scipy.sparse import linalg as sl
from multiprocessing import Pool
from functools import partial


#########################################################################################
#                   Display Mesh
#########################################################################################

def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [ [int(i) for i in thing] for thing in F]
    for i in range(NPs):
        E[:,3*i]=V[:,3*i]
        E[:,3*i+1]=-V[:,3*i+2]
        E[:,3*i+2]=V[:,3*i+1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('Myobj', me)
        scn = bpy.context.scene
        scn.objects.link(ob)
        #scn.objects.active = ob
        #ob.select = True
        
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()

#########################################################################################
#                   Vector Rotation
#########################################################################################

def VecRotation(rotateTowardVec, targetVec):
    # Rotate 'targetVec' towards 'rotateTowardVec'
    w=np.cross(targetVec,rotateTowardVec)
    if np.linalg.norm(w)==0.0:
        R=np.eye(3)
        theta=0
    else:
        w=w/np.linalg.norm(w)
        Dot_prdct=np.dot(rotateTowardVec,targetVec)
        tmp=Dot_prdct/(np.linalg.norm(rotateTowardVec)*np.linalg.norm(targetVec))
        if tmp>1.0:
            theta=0.0
        else:
            theta=np.arccos(tmp)
        
        S=np.sin(theta)
        C=np.cos(theta)
        T=1-C
        R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],[T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],[T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    return R    

#########################################################################################
#                   Face Transformation
#########################################################################################
 
def FaceTransform(TargateTri,SourceTri,NVF):
    # Function gives transformation "T" which transforms "SourceTri" to "TargateTri"
    T=np.zeros([3*NVF,3*NVF])
    if NVF!=1:
        TargateTri=TargateTri-np.mean(TargateTri,axis=0)
        SourceTri=SourceTri-np.mean(SourceTri,axis=0)
    for i in range(NVF):
        R=VecRotation(TargateTri[i,:],SourceTri[i,:])
        V=np.dot(R,SourceTri[i,:].T)
        A=np.linalg.norm(TargateTri[i,:])/np.linalg.norm(V)
        R=A*R
        T[3*i:3*i+3,3*i:3*i+3]=R     
    return T

#############################################################################################
#                       PI Close Form
#############################################################################################
def GetMatrices(In):
    print(In,len(In/2))
    Q=np.reshape(In[0:len(In)/2],(len(In)/6,3))
    T=np.reshape(In[len(In)/2:],(len(In)/6,3))
    temp=FaceTransform(T-np.mean(T,axis=0),Q-np.mean(Q,axis=0),len(Q))
    return np.reshape(temp,np.size(temp))

def ConnectionMatrices(src,trgt,fcs,NV,NF,NVF):
    Phi=sp.lil_matrix((3*NVF*NF,3*NVF*NF))
    B=sp.lil_matrix((3*NVF*NF,3*NV))
    A=sp.lil_matrix((3*NVF*NF,3*NV))
    

    Phi=sp.lil_matrix((3*NVF*NF,3*NVF*NF))
    B=sp.lil_matrix((3*NVF*NF,3*NV))
    A=sp.lil_matrix((3*NVF*NF,3*NV))

    f=np.reshape(fcs,NVF*NF)
    SRC=np.reshape(src[f,:],(NF,3*NVF))
    TRGT=np.reshape(trgt[f,:],(NF,3*NVF))
    InPut=np.concatenate((SRC,TRGT),axis=1)

    p=Pool(4)
    Y=p.map(GetMatrices,InPut)
    FF=np.concatenate((3*np.reshape(fcs[:,0],(NF,1)),3*np.reshape(fcs[:,0],(NF,1))+1,3*np.reshape(fcs[:,0],(NF,1))+2),axis=1)
    for i in range(1,NVF):
        FF=np.concatenate((FF,3*np.reshape(fcs[:,i],(NF,1)),3*np.reshape(fcs[:,i],(NF,1))+1,3*np.reshape(fcs[:,i],(NF,1))+2),axis=1)
    #FF=np.concatenate((3*np.reshape(fcs[:,0],(NF,1)),3*np.reshape(fcs[:,0],(NF,1))+1,3*np.reshape(fcs[:,0],(NF,1))+2,
     #                  3*np.reshape(fcs[:,1],(NF,1)),3*np.reshape(fcs[:,1],(NF,1))+1,3*np.reshape(fcs[:,1],(NF,1))+2,
     #                  3*np.reshape(fcs[:,2],(NF,1)),3*np.reshape(fcs[:,2],(NF,1))+1,3*np.reshape(fcs[:,2],(NF,1))+2),axis=1)
    
    for t in range(NF):
        Phi[3*NVF*t:3*NVF*t+3*NVF,3*NVF*t:3*NVF*t+3*NVF]=np.reshape(np.array(Y[t]),(3*NVF,3*NVF))
        B[3*NVF*t:3*NVF*t+3*NVF,FF[t,:]]=(np.array([[1,0,0]*NVF,[0,1,0]*NVF,[0,0,1]*NVF]*NVF,dtype=float))/NVF
        A[3*NVF*t:3*NVF*t+3*NVF,FF[t,:]]=np.eye(3*NVF)

    M=Phi.dot(A-B)+B
    InvMltply=sp.diags((1.0/A.sum(0)).tolist()[0],0)
    PsdA=(A.dot(InvMltply)).transpose()

    return PsdA.dot(M)
#####################################

def PICloseForm(P,TInpt,F):    
    NF,NVF=np.shape(F)
    NV,NPs=np.shape(P)
    NV=int(NV/3)
 
    S=np.zeros([3*NV,NPs])
    TrNPs=np.size(TInpt)/(3*NV)
    if TrNPs==1:
        S=np.zeros([3*NV,NPs+1])
        S[:,0],S[:,NPs]=1*TInpt,1*TInpt
        P=np.append(P,np.reshape(P[:,0],(3*NV,1)),axis=1) 
        NPs+=1
        TrNPs+=1       
    else:
        S[:,0],S[:,NPs-1]=TInpt[:,0],TInpt[:,TrNPs-1]
        

    print(P,S)
    P1=np.reshape(P[:,0],[NV,3])
    S1=np.reshape(S[:,0],[NV,3])
    
    print ('get connection matrices ....')
    X=ConnectionMatrices(P1,S1,F,NV,NF,NVF)
   
    PR=np.roll(P,1,axis=1)
    PL=np.roll(P,-1,axis=1)
    grd=X.dot(2*P-PR-PL)


    w=1.9
    for itr in range(100):
        for ps in range(1,NPs-1):
            S[:,ps]=(1-w)*S[:,ps]+(w/2)*(grd[:,ps]+S[:,ps-1]+S[:,ps+1])
    
    V_out=np.zeros([NV,3*(NPs-TrNPs)])
    
    for i in range(int(NPs-TrNPs)):
        V_out[:,3*i:3*i+3]=np.reshape(S[:,i+1],[NV,3])
    
    CreateMesh(V_out,F,int(NPs-TrNPs))

    return
    
#############################################################################################
#                       Sumner and Popovic
#############################################################################################

def GetL(InFace):
    L=np.zeros([3,3])
    InFace=np.reshape(InFace,(3,3))-InFace[0:3]
    #print np.shape(InFace[1:,:].T)
    Q,R=np.linalg.qr(InFace[1:,:].T)
    L[:,1:]=np.dot(np.linalg.inv(R),Q.T).T
    L[:,0]=-L[:,1]-L[:,2]
    return L

def GetY(InFace):
    S1=np.reshape(InFace[0:9],[3,3])-InFace[0:3]
    N=np.cross(S1[1,:],S1[2,:])
    N=N/np.linalg.norm(N)
    X=np.concatenate((S1[1:,:],np.array([N.tolist()])),axis=0)
    S2=np.reshape(InFace[9:],[3,3])-InFace[9:12]
    N=np.cross(S2[1,:],S2[2,:])
    N=N/np.linalg.norm(N)
    Y=np.concatenate((S2[1:,:],np.array([N.tolist()])),axis=0)
    Q,R=np.linalg.qr(X.T)
    tmp=np.dot(np.linalg.inv(R),Q.T)
    return np.dot(Y.T,tmp).T

def DTSumAndPop(sourceInpt,TrgtInpt,F):

    NV=np.size(TrgtInpt)/3
    NF,NVF=np.shape(F)

    S=np.zeros([NV,6])
    S[:,0:3]=np.reshape(sourceInpt[:,0],[NV,3])                
    S[:,3:6]=np.reshape(sourceInpt[:,1],[NV,3])

    P=np.zeros([NV,6])
    P[:,0:3]=np.reshape(TrgtInpt,[NV,3])


    A=sp.lil_matrix((3*NF,NV))
    p=Pool()
    fcs=np.reshape(F,NVF*NF)
    PrllIn=np.reshape(P[fcs,0:3],(NF,NVF**2))
    PrllOut=p.map(GetL,PrllIn)

    for t in range(len(F)):
        A[3*t:3*t+3,F[t]]=PrllOut[t]
    p.close()
    A=A.tocsc()
             
    c=sp.lil_matrix(P[NV-1,0:3])
    print(c)

    Y=sp.lil_matrix((3*NF,3))

    p=Pool()
    fcs=np.reshape(F,NVF*NF)
    PrllIn=np.concatenate((np.reshape(S[fcs,0:3],(NF,NVF**2)),
                          np.reshape(S[fcs,3:6],(NF,NVF**2))),axis=1)
    PrllOut=p.map(GetY,PrllIn)

    for t in range(len(F)):
        Y[3*t:3*t+3,:]=PrllOut[t]
    p.close()

    Y=Y-A[:,NV-1].dot(c)
    A=A.tocsc()
    tmp=A[:,:(NV-1)].transpose()
    A=tmp.dot(tmp.transpose())
    b=tmp.dot(Y)
    P[:NV-1,3]=sl.spsolve(A,b[:,0])
    P[:NV-1,4]=sl.spsolve(A,b[:,1])
    P[:NV-1,5]=sl.spsolve(A,b[:,2])

    P[NV-1,3:6]=c.toarray()
    CreateMesh(P[:,3:6],F,1)
    return

#####################################################################################################
#             Semantic Deformation Transfer
#####################################################################################################
def LogRotation(RotMat):
       
    if ((np.trace(RotMat)-1)/2)>=1:
        Theta=np.arccos(1)+0.001
    elif ((np.trace(RotMat)-1)/2)<=-1:
        Theta=np.arccos(-1)-0.001
    else:
        Theta=np.arccos((np.trace(RotMat)-1)/2)         
    Axs=np.array([RotMat[2,1]-RotMat[1,2],RotMat[0,2]-RotMat[2,0],RotMat[1,0]-RotMat[0,1]])/(2*np.sin(Theta))
    return Theta*Axs

def ExpRotation(vec):
    angl=np.sqrt(np.sum(vec**2))
    axs=vec/angl
    K=np.array([[0,-axs[2],axs[1]],[axs[2],0,-axs[0]],[-axs[1],axs[0],0]])
    RotMat=np.eye(3)+np.sin(angl)*K+(1-np.cos(angl))*(K.dot(K))
    return RotMat

def FaceNeighbor(F):
    NgbrF=[]
    t=0
    for f in F:
        temp=[]
        for i in f:
            u=np.where((F-i)==0)[0]
            temp=temp+u.tolist()
        temp3=np.array(list(set(temp)))
        temp2=[temp.count(i) for i in temp3.tolist()]
        NgbrF.insert(t,temp3[np.where(np.array(temp2)==2)[0]].tolist())
        t+=1
    return NgbrF

def getS(TempIn):
    Rest=np.reshape(TempIn[0:9],(3,3))-TempIn[0:3]
    Current=np.reshape(TempIn[9:],(3,3))-TempIn[9:12]

    Rest[0,:]=np.cross(Rest[1,:],Rest[2,:])/np.sqrt(np.sum(np.cross(Rest[1,:],Rest[2,:])**2))
    Rest=np.roll(Rest.T, -1, axis=1)

    Current[0,:]=np.cross(Current[1,:],Current[2,:])/np.sqrt(np.sum(np.cross(Current[1,:],Current[2,:])**2))
    Current=np.roll(Current.T, -1, axis=1)
    D=Rest.dot(np.linalg.inv(Current))
    
    q, r = np.linalg.qr(D)
    
    return q, r.dot(np.ones(3))

def GetAxis(QQ,NGBR):
    Ax=np.zeros(9)
    for j in range(3):
            temp=np.linalg.inv(Q[3*NgbrF[i][j]:3*NgbrF[i][j]+3,:])
            Ax[3*j:3*j+3]=LogRotation(temp.dot(Q[3*i:3*i+3,:]))
    return Ax

def ShapeSpace(RestPose,CurrentPose,F,NgbrF):    
    Q=np.zeros([len(F)*3,3])
    XX=np.zeros(len(F)*12+6)
    
    p=Pool()
    f=np.reshape(F,np.size(F))
    prllIn=np.concatenate(( np.reshape(RestPose[f,:],(len(F),9)),
                            np.reshape(CurrentPose[f,:],(len(F),9)) ),axis=1)
    
    prllOut=p.map(getS,prllIn)
    p.close()
    
    temp=0
    for t in range(len(F)):
        temp+= prllOut[t][0]
        Q[3*t:3*t+3,:]=prllOut[t][0]
        XX[3*t:3*t+3]=prllOut[t][1]

    XX[3*len(F):3*len(F)+3]=np.mean(CurrentPose,axis=0)
    Qbar, r = np.linalg.qr(temp)
    XX[3*len(F)+3:3*len(F)+6]=LogRotation(Qbar)
    

    for i in range(len(F)):
        for j in range(3):
            temp=np.linalg.inv(Q[3*NgbrF[i][j]:3*NgbrF[i][j]+3,:])
            XX[3*len(F)+6+9*i+3*j:3*len(F)+6+9*i+3*j+3]=LogRotation(temp.dot(Q[3*i:3*i+3,:]))
    
    return XX

def Reconstruction(YY,F,NgbrF,XX):
    NV=YY[0]
    NF=YY[1]
    RestPose= np.reshape(YY[2:3*NV+2],(6890,3))
    F=np.reshape(F,(NF,3))

    S=XX[0:3*len(F)]
    Vbar=XX[3*len(F):3*len(F)+3]
    Qbar=ExpRotation(XX[3*len(F)+3:3*len(F)+6])
    Axis=XX[3*len(F)+6:,]
    

    B=sp.lil_matrix((9*len(F),3*len(F)))
    A=sp.lil_matrix((9*len(F),3*len(F)))
    Phi=sp.lil_matrix((9*len(F),9*len(F)))
    Y=sp.lil_matrix((3*len(F),3*len(F)))
    Y[0:3,0:3]=np.eye(3)
    YY=sp.lil_matrix((3*len(RestPose),3*len(RestPose)))
    YY[0:3,0:3]=np.eye(3)
    L=np.array([[1,0,0,-1,0,0,0,0,0],[0,1,0,0,-1,0,0,0,0],[0,0,1,0,0,-1,0,0,0],
                [0,0,0, 1,0,0,-1,0,0],[0,0,0,0,1,0,0,-1,0],[0,0,0,0,0,1,0,0,-1],
                [-1,0,0,0,0,0,1,0,0],[0,-1,0,0,0,0,0,1,0],[0,0,-1,0,0,0,0,0,1]])
    AA=sp.lil_matrix((9*len(F),3*len(RestPose)))

    for t in range(len(F)):    
        A[9*t:9*t+9,3*t:3*t+3]=(np.array([[1,0,0],[0,1,0],[0,0,1]]*3,dtype=float))
        AA[9*t:9*t+9,3*F[t][0]:3*F[t][0]+3]=L[:,0:3]
        AA[9*t:9*t+9,3*F[t][1]:3*F[t][1]+3]=L[:,3:6]
        AA[9*t:9*t+9,3*F[t][2]:3*F[t][2]+3]=L[:,6:9]
        l=0
        for i in NgbrF[t]:
            B[9*t+3*l:9*t+3*l+3,3*i:3*i+3]=np.eye(3)
            temp=ExpRotation(Axis[9*t+3*l:9*t+3*l+3])
            Phi[9*t+3*l:9*t+3*l+3,9*t+3*l:9*t+3*l+3]=temp.T
            l+=1       
    X=A-Phi.dot(B)
    H=X.transpose().dot(X)+Y.transpose().dot(Y)
    b=sp.lil_matrix((3*len(F),3))
    b[0:3,0:3]=np.eye(3)
    b=Y.transpose().dot(b)
    G=sl.spsolve(H.tocsc(),b.tocsc())

    V_tilde=np.zeros(9*len(F))
    for t in range(len(F)):
        q,r=np.linalg.qr(G[3*t:3*t+3,:].toarray())
        DD=q.dot(np.diag(S[3*t:3*t+3]))
        V_tilde[9*t:9*t+3]=DD.dot(RestPose[F[t][0],:]-RestPose[F[t][1],:])
        V_tilde[9*t+3:9*t+6]=DD.dot(RestPose[F[t][1],:]-RestPose[F[t][2],:])
        V_tilde[9*t+6:9*t+9]=DD.dot(RestPose[F[t][2],:]-RestPose[F[t][1],:])
        

    XX=AA.transpose().dot(AA)+YY.transpose().dot(YY)
    bb=AA.transpose().dot(V_tilde)
    V_desh=sl.spsolve(XX.tocsc(),bb)
    V_desh=np.reshape(V_desh,[len(RestPose),3])
    
    temp=0
    for f in F:
        v1=RestPose[f[1],:]-RestPose[f[0],:]
        v2=RestPose[f[2],:]-RestPose[f[0],:]
        nv=np.cross(v1,v2)/np.sqrt(np.sum(np.cross(v1,v2)**2))
        temp1=np.zeros([3,3])
        temp1[:,0]=v1
        temp1[:,1]=v2
        temp1[:,2]=nv

        u1=V_desh[f[1],:]-V_desh[f[0],:]
        u2=V_desh[f[2],:]-V_desh[f[0],:]
        nu=np.cross(u1,u2)/np.sqrt(np.sum(np.cross(u1,u2)**2))
        temp2=np.zeros([3,3])
        temp2[:,0]=u1
        temp2[:,1]=u2
        temp2[:,2]=nu
        D=temp1.dot(np.linalg.inv(temp2))
        q, r = np.linalg.qr(D)
        temp+=q
        
    VBarDesh=np.mean(V_desh,axis=0)
    V=V_desh-VBarDesh+Vbar

    return V

def DTSemantic(Src,Trgt,F):
    
    NgbrF=FaceNeighbor(F)
    NExmpl=len(Trgt[0,:])
    SSsrc=np.zeros([12*len(F)+6,NExmpl])
    SStrgt=np.zeros([12*len(F)+6,NExmpl])
    
    for i in range(NExmpl):
        SSsrc[:,i]=ShapeSpace(np.reshape(Src[:,0],(len(Src)/3,3)),np.reshape(Src[:,i],(len(Src)/3,3)),F,NgbrF)
        SStrgt[:,i]=ShapeSpace(np.reshape(Trgt[:,0],(len(Trgt)/3,3)),np.reshape(Trgt[:,i],(len(Trgt)/3,3)),F,NgbrF)
    X=[]
    for m in range(NExmpl,len(Src[0,:])):
        SSsrcInput=ShapeSpace(np.reshape(Src[:,0],(len(Src)/3,3)),np.reshape(Src[:,m],(len(Src)/3,3)),F,NgbrF)
        W=(np.linalg.pinv(SSsrc)).dot(SSsrcInput)
        SSsrc=np.concatenate((SSsrc,np.reshape(SSsrcInput,(len(SSsrcInput),1))),axis=1)
        temp=SStrgt.dot(W.T)
        SStrgt=np.concatenate((SStrgt,np.reshape(temp,(len(temp),1))),axis=1)
        X.append(temp)
        
    p=Pool()
    FixArg=np.concatenate((np.array([len(Src)/3]),
                           np.array([len(F)]),
                           Trgt[:,0]))

    func=partial(Reconstruction,FixArg,np.reshape(F,np.size(F)),NgbrF)
    Y=p.map(func,np.array(X))
    
    for i in range(len(Src[0,:])-NExmpl):     
        CreateMesh(np.reshape(Y[i],[len(Trgt)/3,3]),F,1)

    return

####################################################################################################
#                                    DT Manifold
####################################################################################################
def CanonicalForm(T):  
    C=np.array([np.linalg.norm(T[1,:]),0,0])
    R1=VecRotation(C,T[1,:])
    p2=np.dot(R1,T[2,:])

    if p2[1]==0 and p2[2]!=0:
        c,s=1.0,0
    else:
        c=p2[1]/np.linalg.norm(p2[1:])
        s=-p2[2]/np.linalg.norm(p2[1:])

    R2=np.array([[1,0,0],[0,c,-s],[0,s,c]])

    return np.dot(R2,R1)

def GetScaleAndTransform(T1,T2):
    
    S=abs(T2[1,0]/T1[1,0])
    A=np.array([[1,(T2[2,0]-T1[2,0])/T1[2,1],0],[0,T2[2,1]/T1[2,1],0],[0,0,1]])
    return S*A


def GetDeformation(In):
    TS0=np.reshape(In[0:9],(3,3))
    TS1=np.reshape(In[9:18],(3,3))
    TT0=np.reshape(In[18:],(3,3))
    RS0=CanonicalForm(TS0-TS0[0,:])
    RS1=CanonicalForm(TS1-TS1[0,:])
    RT0=CanonicalForm(TT0-TT0[0,:])
    B=np.dot(RT0,RS0.T)
    C=np.dot(RS1.T,B.T)
    D=GetScaleAndTransform(np.dot(RS0,(TS0-TS0[0,:]).T).T,np.dot(RS1,(TS1-TS1[0,:]).T).T)
    E=np.dot(RT0,(TT0-TT0[0,:]).T)
    return np.dot(np.dot(C,D),E).T


def DTManifold(Src,Trgt,F):

    S0=np.reshape(Src[:,0],(len(Src)/3,3))
    S1=np.reshape(Src[:,1],(len(Src)/3,3))
    P0=np.reshape(Trgt,(len(Trgt)/3,3))

    NV,nop=np.shape(S0)
    NF,NVF=np.shape(F)
    NPs=11

    A=sp.lil_matrix((3*NF,NV))
    
    for t in range(len(F)):
        A[3*t:3*t+3,F[t]]=np.array([[0,0,0],[-1,1,0],[-1,0,1]])
    A=A.tocsc()
    tmp=A[:,0:(NV-1)].transpose()
    B=tmp.dot(tmp.transpose())
    
    c=sp.csc_matrix(P0[NV-1,:])

    
    b=np.zeros([3*NF,3])
    fcs=np.reshape(F,NF*NVF)
    p=Pool()
    PrllIn=np.concatenate((np.reshape(S0[fcs,:],(NF,9)),
                           np.reshape(S1[fcs,:],(NF,9)),
                           np.reshape(P0[fcs,:],(NF,9))
                           ),axis=1)
    PrllOut=p.map(GetDeformation,PrllIn)
    
    for t in range(len(F)):
        b[3*t:3*t+3,:]=PrllOut[t]
    p.close()    
        
    Y=np.array(b-A[:,NV-1].dot(c))
    b=tmp.dot(Y)
    
    P1=np.zeros([NV,3])
    P1[:NV-1,0]=sl.spsolve(B,b[:,0])
    P1[:NV-1,1]=sl.spsolve(B,b[:,1])
    P1[:NV-1,2]=sl.spsolve(B,b[:,2])
    P1[NV-1,:]=c.toarray()
    CreateMesh(P1,F,1)
    return 

    



#####################################################################################################
#           Deformation Transfer
#####################################################################################################

class DTToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
 
    def draw(self, context):
        self.layout.operator("get.seq",text='Source Seq').seqType="source"
        self.layout.operator("get.seq",text='Target Seq').seqType="target" 
        self.layout.operator("dt.tools",text='DTSumnerPopovic').seqType="DTSumnerPopovic"
        self.layout.operator("dt.tools",text='DTSemantic').seqType="DTSemantic"
        self.layout.operator("dt.tools",text='DTManifold').seqType="DTManifold" 
        self.layout.operator("dt.tools",text='DTPI').seqType="DTPI" 
        

# Operator
class GetSequence(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        path=bpy.utils.resource_path('USER')
        print(path)
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        F=np.zeros([len(obj.data.polygons),len(obj.data.polygons[0].vertices)],dtype=int)
        V=np.zeros([3*len(obj.data.vertices),len(Selected_Meshes)])
        t=0
        for f in obj.data.polygons:
                F[t,:]=f.vertices[:]
                t+=1
        
        for i in range(len(Selected_Meshes)):
            bpy.context.scene.objects.active = Selected_Meshes[-i-1]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world*v.co
                V[3*t:3*t+3,i]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
        np.savetxt(path+self.seqType+'_vertz.txt',V,delimiter=',')
        np.savetxt(path+'facez.txt',F,delimiter=',')                      
        return{'FINISHED'}    
 

class DeformationTransferTools(bpy.types.Operator):
    bl_idname = "dt.tools"
    bl_label = "DT Tools"
    seqType = bpy.props.StringProperty()
    def execute(self,context):
        path=bpy.utils.resource_path('USER') 
        
        sourceInpt=np.loadtxt(path+'source_vertz.txt',delimiter=',')
        TrgtInpt=np.loadtxt(path+'target_vertz.txt',delimiter=',')
        F=np.loadtxt(path+'facez.txt',delimiter=',').astype(int)
        
        if self.seqType=='DTSumnerPopovic':
            DTSumAndPop(sourceInpt[:,0:2],TrgtInpt,F)
        elif self.seqType=='DTSemantic':
            DTSemantic(sourceInpt,TrgtInpt,F)
        elif self.seqType=='DTManifold':
            DTManifold(sourceInpt[:,0:2],TrgtInpt,F)    
        else:
            PICloseForm(sourceInpt,TrgtInpt,F)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(DTToolsPanel)
    bpy.utils.register_class(GetSequence)
    bpy.utils.register_class(DeformationTransferTools)
    
   

def unregister():
    bpy.utils.unregister_class(DTToolsPanel)
    bpy.utils.unregister_class(GetSequence)
    bpy.utils.unregister_class(DeformationTransferTools)
 
if __name__ == "__main__":  # only for live edit.
    bpy.utils.register_module(__name__) 

