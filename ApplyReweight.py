#!/usr/bin/env python
# coding: utf-8

# # Demonstration of distribution reweighting
# 
#     requirements:
#     xgboost
#     numpy
#     matplotlib
#     sklearn
#     pandas
#     ROOT 6.12
#     root_numpy (using python 2.7, download root_numpy from official web and install by hand)

# In[1]:
#version 1 in github issue with noly B pt and all significance divide into value and error

from __future__ import division
#get_ipython().run_line_magic('pylab', 'inline')
#figsize(16, 8)
import xgboost as xgb 
from scipy.stats import ks_2samp
from xgboost import plot_importance
from xgboost import plot_tree
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn import metrics
import pandas as pd
import root_numpy
from array import array
import ROOT as r
from ROOT import TCut
import time as timer
import sys
from  sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, roc_curve

r.gROOT.SetBatch(True)

time_start=timer.time()

#number = 100000
q2Bin = int(sys.argv[1])
parity = int(sys.argv[2])
year = int(sys.argv[3])
# In[2]:


variables = {} #dictionary containing all the variables

class variable:

    def __init__(self, name, title, binning=None):
        if binning is not None and (not isinstance(binning, list) or len(binning)!=3):
            raise Exception("Error in declaration of variable {0}. Binning must be a list like [nbins, min, max]".format(name))
        self.title = title
        self.binning = binning
        variables[name] = self #add yourself to dictionary

    def get_nbins(self):
        if self.binning is not None:
            return self.binning[0]
        else: return 50

    def get_xmin(self):
        if self.binning is not None:
            return self.binning[1]
        else: return 0

    def get_xmax(self):
        if self.binning is not None:
            return self.binning[2]
        else: return 0

#declare variables

#variable of BDT
variable("kstarmass", "kstar tagged mass", [100, 0.74, 1.05 ])
variable("bVtxCL","B decay vertex CL",[100, 0.0, 1.0])
variable("bLBS"," bLBS ",[100, 0, 1.5])
variable("bLBSE_scaled"," bLBSE_scaled ",[100, 0, 0.04]) #significance
variable("bLBSsig_scaled"," bLBSsig_scaled ",[100, 0, 200])

variable("bCosAlphaBS"," bCosAlphaBS ",[100, 0.994, 1.0])

variable("bDCABS","bDCABS",[100, -0.015, 0.015])
variable("bDCABSE_scaled","bDCABSE_scaled",[100, 0, 0.01]) #significance
variable("bDCABSsig_scaled","bDCABSsig_scaled",[100, -5, 5])

variable("kstTrk1DCABS","kstTrk1DCABS",[100, -1, 1])
variable("kstTrk1DCABSE_scaled"," kstTrk1DCABSE_scaled",[100, 0, 0.025]) #significance
variable("kstTrk1DCABSsig_scaled"," kstTrk1DCABSsig_scaled",[100, -50,50])

variable("kstTrk2DCABS","kstTrk2DCABS",[100, -1, 1])
variable("kstTrk2DCABSE_scaled"," kstTrk2DCABSE_scaled",[100, 0, 0.025]) #significance
variable("kstTrk2DCABSsig_scaled"," kstTrk2DCABSsig_scaled",[100, -50,50])

variable("kstTrk2MinIP2D","kstTrk2MinIP2D ",[100, 0, 1])
variable("kstTrk1MinIP2D","kstTrk1MinIP2D ",[100, 0, 1])

variable("sum_isopt_04","sum_isopt_04 ",[100, 0, 40])


#others draw picture from chuqiao

variable("mumPt", "mumPT", [100,0,40])
variable("mupPt", "mupPT", [100,0,40])

variable("mumEta", "mumEta", [100,-2.5,2.5])
variable("mupEta", "mupEta", [100,-2.5,2.5])

variable("mumPhi", "mumPhi", [100,-3.14159,3.14159])
variable("mupPhi", "mupPhi", [100,-3.14159,3.14159])

#variable("mumuLBS", "mumuLBS", [100,0,6.4])
#variable("mumuLBSE", "mumuLBSE", [100,0,0.12])

#variable("mumuDCA", "mumuDCA", [100,0,1])

#variable("mumuCosAlphaBS", "mumuCosAlphaBS", [100,0.93,1])

variable("kstTrk1Pt", "kstTrk1Pt", [100,0,20])
variable("kstTrk2Pt", "kstTrk2Pt", [100,0,20])

variable("kstTrk1Eta", "kstTrk1Eta", [100,-2.5,2.5])
variable("kstTrk2Eta", "kstTrk2Eta", [100,-2.5,2.5])

variable("kstTrk1Phi", "kstTrk1Phi", [100,-3.14159,3.14159])
variable("kstTrk2Phi", "kstTrk2Phi", [100,-3.14159,3.14159])

variable("bPt","bPt",[100,0,80])
variable("bEta", "bEta", [100,-2.5,2.5])
variable("bPhi", "bPhi", [100,-3.14159,3.14159])

variable("tagged_mass", "tagged_mass", [100,5.0,5.6])

variable("cos_theta_l", "cos_theta_l", [100,-1,1])
variable("cos_theta_k", "cos_theta_k", [100,-1,1])
variable("phi_kst_mumu", "phi_kst_mumu", [100,-3.14159,3.14159])
variable("kstVtxCL","kstVtxCL",[100,0,1])


variable("weight","PUweight",[100,-1,3])

rdata = r.TChain("ntuple")
rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/recoDATADataset_b{}_{}_p{}_aftersel.root'.format(q2Bin,year,parity))
MC = r.TChain("ntuple")
MC.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}_MC_JPSI_scale_and_preselection_p{}.root'.format(year,parity))
MC_friend = r.TTree("wTree", "weights tree")
leafValues = array("f", [0.0])
weight_branch = MC_friend.Branch("MCw", leafValues,"MCw[1]/F")

print("Wtree builded")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Training samples preparation------------------------------------------")
columns = ['bVtxCL', 'bLBS', 'bLBSE' ,'bEta','bCosAlphaBS', 'bDCABS','bDCABSE', 'kstTrk1DCABS','kstTrk1DCABSE','kstTrk2DCABS','kstTrk2DCABSE','sum_isopt_04']
sw_branch = ['nsig_sw']
weight_branch = ['weight']

data_ori = root_numpy.tree2array(rdata,branches=columns)
print("Data sample readed------------------------------------------")
phsp_ori = root_numpy.tree2array(MC,branches=columns)
print("MC sample readed------------------------------------------")
JpsiKSignal_SW = root_numpy.tree2array(rdata,branches=sw_branch)

print("Sweights readed------------------------------------------")
MCPUweight = root_numpy.tree2array(MC,branches=weight_branch)
MCPUweight=MCPUweight.reshape(-1,1).astype(float)
MCweight = MCPUweight

data_only_X=pd.DataFrame(data_ori,columns=columns)
phsp_only_X=pd.DataFrame(phsp_ori,columns=columns)
corrdata = data_only_X.corr()
#plt.figure(dpi=400,figsize=(20,20))
'''plt.figure().set_size_inches(10.5, 9.5)
datamap = sb.heatmap(corrdata.round(2), cmap="Blues", annot=True)
datafig = datamap.get_figure()
datafig.savefig("./plots/datacor{}_v4.png".format(year))
plt.clf()
corrMC = phsp_only_X.corr()
MCmap = sb.heatmap(corrMC.round(2), cmap="Blues", annot=True)
MCfig = MCmap.get_figure()
MCfig.savefig("./plots/MCcor{}_v4.png".format(year))
sys.exit(0)
'''
sw_sig_RD_X=pd.DataFrame(JpsiKSignal_SW)
data_signal_sumEntries = (sw_sig_RD_X.sum())[0]

w_MC_RD_X = pd.DataFrame(MCweight)
MC_signal_sumEntries = (w_MC_RD_X.sum())[0]
print("data_signal_sumEntries = {0}",(sw_sig_RD_X.sum())[0])
print("MC_sumEntries = {0}",(w_MC_RD_X.sum())[0])
sf_phsp_vs_data=(MC_signal_sumEntries)/(data_signal_sumEntries)
print("scale factor of MC/data is", sf_phsp_vs_data)

#Make labels for each data, data marks as 1, MC marks as 0 
data_only_Y=np.ones(len(data_only_X))
phsp_only_Y=np.zeros(len(phsp_only_X))

data=pd.DataFrame(data_ori)
phsp=pd.DataFrame(phsp_ori)


data_only_a=np.array(data)
phsp_only_a=np.array(phsp)
sw_sig_RD_a=np.array(sw_sig_RD_X)
sw_sig_RD_a=sw_sig_RD_a.reshape(-1,1)
w_MC_a = np.array(w_MC_RD_X)
w_MC_a = w_MC_a.reshape(-1,1)

if((data_only_a.shape)[0] != (sw_sig_RD_a.shape)[0]): 
    print ('shapes of data and sweight are not the same')

from sklearn.model_selection import train_test_split
data_all = np.concatenate([data_only_a, phsp_only_a], axis=0)
sw_sig_RD_a = sf_phsp_vs_data * sw_sig_RD_a
sweights_all = np.concatenate([sw_sig_RD_a, w_MC_a], axis=0) 
data_all = np.concatenate([data_all, sweights_all], axis=1)
labels_all = np.array([1] * len(data_only_a) + [0] * len(phsp_only_a))

train_X,test_X,train_Y,test_Y=train_test_split(data_all,labels_all,test_size=0.1,stratify=labels_all)
sw_train = train_X[:,-1]
sw_train = sw_train.tolist()
sw_test = test_X[:,-1]
sw_test = sw_test.tolist()
train_X = train_X[:,:-1]
test_X = test_X[:,:-1]

xg_train = xgb.DMatrix(train_X, label=train_Y, weight=sw_train)
xg_test = xgb.DMatrix(test_X, label=test_Y, weight=sw_test)
xg_data_only = xgb.DMatrix(data_only_X, label=data_only_Y, weight=(sw_sig_RD_a))
xg_phsp_only = xgb.DMatrix(phsp_only_X, label=phsp_only_Y, weight=(w_MC_a))

Save_Dir = './nobPt_{}_XGBV4_eta5_subsample5_depth4_round200.json'.format(year)
trained_bst = xgb.Booster(model_file=Save_Dir)

pr_phsp=np.array(trained_bst.predict(xg_phsp_only,validate_features=False).reshape(phsp_only_Y.shape[0], 2))

#weight_test=pr_test[:,1]/pr_test[:,0]
weight_phsp=pr_phsp[:,1]/pr_phsp[:,0]
print("MC_weights: weight_phsp=pr_phsp")
print(weight_phsp)

# In[11]:

print("MC weights------------------------------------------")

MCwFile = r.TFile("./JPsiK_reweight_nobPt_{}.root".format(year),"RECREATE")

for val in weight_phsp:
    leafValues[0] = val
    MC_friend.Fill()

MCwFile.cd()
MC_friend.Write()

MC.AddFriend(MC_friend)

time_end=timer.time()
print('totally cost',time_end-time_start)