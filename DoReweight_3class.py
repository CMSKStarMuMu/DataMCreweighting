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
eta = int(sys.argv[4])
subsample = int(sys.argv[5])
max_depth = int(sys.argv[6])
num_round = int(sys.argv[7])

eta1 = float(eta)/100.
subsample1 = float(subsample)/10.

print ('eta ', eta1, ' subsample ', subsample1, ' max_depth ', max_depth, ' num_round ', num_round)

variables = {} #dictionary containing all the variables
mc_sigma = 0.040
mc_mass  = 5.27783 
JPsiMass_ = 3.096916
nSigma_psiRej = 3.
selData = '( pass_preselection ==1 ) && \
            (abs(mumuMass - {JPSIM}) < {CUT}*mumuMassE) &&  eventN%2=={Parity}'\
            .format( JPSIM=JPsiMass_, CUT=nSigma_psiRej, Parity=parity)

selData1 = '( pass_preselection ==1 ) && \
            (abs(mumuMass - {JPSIM}) < {CUT}*mumuMassE) &&  eventN_x%2=={Parity}'\
            .format( JPSIM=JPsiMass_, CUT=nSigma_psiRej, Parity=parity)

selMC = selData1 + ' && (trig==1)' + '&& truthMatchMum == 1 && truthMatchMup == 1 && truthMatchTrkm == 1 && truthMatchTrkp == 1'
#selNan = '  && bLBSE==bLBSE && bDCABSE==bDCABSE && abs(bCosAlphaBS)<=1 && sum_isopt_04<10'
selPosdata = ' && nsig_sw>0'
selNegdata = ' && nsig_sw<=0'
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

variable("mu1Pt", "mu1PT", [100,0,40])
variable("mu2Pt", "mu2PT", [100,0,40])

variable("mu1Eta", "mu1Eta", [100,-2.5,2.5])
variable("mu2Eta", "mu2Eta", [100,-2.5,2.5])

variable("mu1Phi", "mu1Phi", [100,-3.14159,3.14159])
variable("mu2Phi", "mu2Phi", [100,-3.14159,3.14159])

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
rdata.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/after_nominal_selection/jpsi_channel_splot/{}data_noIP2D_addxcutvariable_passSPlotCuts_mergeSweights.root".format(year))
MC = r.TChain("ntuple")
MC.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGB_postBDT/{}MC_JPSI_forXGB_AddDRweight.root".format(year))
#MC.Add("/eos/user/a/aboletti/BdToKstarMuMu/fileIndex/MC-Jpsi-presel-scaled/{}.root".format(year))
#rDCrate = r.TChain("ntuple")
#rDCrate.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_MC_JPSI_scale_and_preselection_p{}.root".format(year,year,parity))
MC_friend = r.TTree("wTree", "weights tree")
leafValues = array("f", [0.0])
weight_branch = MC_friend.Branch("MCw", leafValues,"MCw[1]/F")

print("Wtree builded")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Training samples preparation------------------------------------------")
#columns = ['bVtxCL', 'bLBS', 'bLBSE' ,'bPt','bEta','bCosAlphaBS', 'bDCABS','bDCABSE', 'kstTrk1DCABS','kstTrk1DCABSE','kstTrk2DCABS','kstTrk2DCABSE','sum_isopt_04']

columns = ['kstTrk1Pt', 'kstTrk2Pt','kstTrk1Eta', 'kstTrk2Eta','mu1Pt','mu2Pt','mu1Eta','mu2Eta']
sw_branch = ['nsig_sw']
weight_branch = ['weight']
DCratew_branch = ['DRweight']

data_pos = root_numpy.tree2array(rdata,branches=columns,selection=selData+selPosdata)
print("Data pos sample readed------------------------------------------", data_pos.shape)

data_neg = root_numpy.tree2array(rdata,branches=columns,selection=selData+selNegdata)
print("Data neg sample readed------------------------------------------", data_neg.shape)


phsp_ori = root_numpy.tree2array(MC,branches=columns,selection=selMC)
print("MC sample readed------------------------------------------" , phsp_ori.shape)

JpsiKSignal_posSW = root_numpy.tree2array(rdata,branches=sw_branch,selection=selData+selPosdata)
JpsiKSignal_posSW =JpsiKSignal_posSW.reshape(-1,1).astype(float)
print("dataposSweights readed------------------------------------------",JpsiKSignal_posSW.shape)


JpsiKSignal_negSW = root_numpy.tree2array(rdata,branches=sw_branch,selection=selData+selNegdata)
JpsiKSignal_negSW =JpsiKSignal_negSW.reshape(-1,1).astype(float)
print("datanegSweights readed------------------------------------------",JpsiKSignal_negSW.shape)
JpsiKSignal_negSW = -JpsiKSignal_negSW 

MCPUweight = root_numpy.tree2array(MC,branches=weight_branch,selection=selMC)
MCPUweight=MCPUweight.reshape(-1,1).astype(float)
print("MCPUweights readed------------------------------------------",MCPUweight.shape)
MCDCratew = root_numpy.tree2array(MC,branches=DCratew_branch,selection=selMC)
MCDCratew =MCDCratew.reshape(-1,1).astype(float)
print("MCDCratew readed------------------------------------------",MCDCratew.shape)
MCweight = MCPUweight*MCDCratew

datapos_only_X=pd.DataFrame(data_pos,columns=columns)
dataneg_only_X=pd.DataFrame(data_neg,columns=columns)
phsp_only_X=pd.DataFrame(phsp_ori,columns=columns)
#corrdatapos = datapos_only_X.corr()
#corrdataneg = dataneg_only_X.corr()
#plt.figure(dpi=400,figsize=(20,20))
'''plt.figure().set_size_inches(20, 22)
datamap = sb.heatmap(corrdata.round(2), cmap="Blues", annot=True)
datafig = datamap.get_figure()
datafig.savefig("./plots/nobPtdatacor{}_v5_noBmass.png".format(year))
plt.clf()
corrMC = phsp_only_X.corr()
MCmap = sb.heatmap(corrMC.round(2), cmap="Blues", annot=True)
MCfig = MCmap.get_figure()
MCfig.savefig("./plots/nobPtMCcor{}_v5_noBmass.png".format(year))
sys.exit(0)
'''

possw_sig_RD_X=pd.DataFrame(JpsiKSignal_posSW)
posdata_signal_sumEntries = (possw_sig_RD_X.sum())[0]

negsw_sig_RD_X=pd.DataFrame(JpsiKSignal_negSW)
negdata_signal_sumEntries = (negsw_sig_RD_X.sum())[0]

data_signal_sumEntries = posdata_signal_sumEntries - negdata_signal_sumEntries

w_MC_RD_X = pd.DataFrame(MCweight)
MC_signal_sumEntries = (w_MC_RD_X.sum())[0]

print("data_signal_sumEntries = {0}",possw_sig_RD_X.sum()[0]-negsw_sig_RD_X.sum()[0])
print("MC_sumEntries = {0}",(w_MC_RD_X.sum())[0])
sf_phsp_vs_data=(MC_signal_sumEntries)/(data_signal_sumEntries)
print("scale factor of MC/data is", sf_phsp_vs_data)

#Make labels for each data, datapos marks as 1,dataneg marks as 2,MC marks as 0 
datapos_only_Y=np.ones(len(datapos_only_X))
dataneg_only_Y=np.full(len(dataneg_only_X),2.)
phsp_only_Y=np.zeros(len(phsp_only_X))

datapos=pd.DataFrame(data_pos)
dataneg=pd.DataFrame(data_neg)
phsp=pd.DataFrame(phsp_ori)


datapos_only_a=np.array(datapos)
dataneg_only_a=np.array(dataneg)
phsp_only_a=np.array(phsp)

possw_sig_RD_a=np.array(possw_sig_RD_X)
possw_sig_RD_a=possw_sig_RD_a.reshape(-1,1)
negsw_sig_RD_a=np.array(negsw_sig_RD_X)
negsw_sig_RD_a=negsw_sig_RD_a.reshape(-1,1)

w_MC_a = np.array(w_MC_RD_X)
w_MC_a = w_MC_a.reshape(-1,1)

#if((data_only_a.shape)[0] != (sw_sig_RD_a.shape)[0]): 
#    print ('shapes of data and sweight are not the same')

from sklearn.model_selection import train_test_split
data_all = np.concatenate([datapos_only_a, dataneg_only_a,phsp_only_a], axis=0)
possw_sig_RD_a = sf_phsp_vs_data * possw_sig_RD_a
negsw_sig_RD_a = sf_phsp_vs_data * negsw_sig_RD_a
sweights_all = np.concatenate([possw_sig_RD_a,negsw_sig_RD_a, w_MC_a], axis=0) 
data_all = np.concatenate([data_all, sweights_all], axis=1)
labels_all = np.array([1] * len(datapos_only_a) + [2] * len(dataneg_only_a)+ [0] * len(phsp_only_a))

train_X,test_X,train_Y,test_Y=train_test_split(data_all,labels_all,test_size=0.1,stratify=labels_all)
sw_train = train_X[:,-1]
sw_train = sw_train.tolist()
sw_test = test_X[:,-1]
sw_test = sw_test.tolist()
train_X = train_X[:,:-1]
test_X = test_X[:,:-1]

xg_train = xgb.DMatrix(train_X, label=train_Y, weight=sw_train)
xg_test = xgb.DMatrix(test_X, label=test_Y, weight=sw_test)
xg_posdata_only = xgb.DMatrix(datapos_only_X, label=datapos_only_Y, weight=(possw_sig_RD_a))
xg_negdata_only = xgb.DMatrix(dataneg_only_X, label=dataneg_only_Y, weight=(negsw_sig_RD_a))
xg_phsp_only = xgb.DMatrix(phsp_only_X, label=phsp_only_Y, weight=(w_MC_a))


params = {}
params['eval_metric'] = 'mlogloss'
# use softmax multi-class classification
#params['objective'] = 'multi:softmax'
# scale weight of positive examples
params['eta'] = eta1
params['max_depth'] = max_depth
params['subsample'] = subsample1
params['silent'] = 1
params['nthread'] = 4
params['num_class'] = 3

params['objective'] = 'multi:softprob'
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = num_round # 迭代次数
evals_result={}
print("training==================================")
bst = xgb.train(params, xg_train, num_round, watchlist,evals_result=evals_result,verbose_eval=True,early_stopping_rounds=10)


plt.clf()
#plt.figure().set_size_inches(10.5, 9.5)
print (evals_result)
auc_train = list(evals_result['train'].values())[0]
auc_test  = list(evals_result['test'].values())[0]

n_estimators = np.arange(len(auc_train))

plt.plot(n_estimators, auc_train, color='r', label='mlogloss train')
plt.plot(n_estimators, auc_test , color='b', label='mlogloss test' )

plt.xlabel('# tree')
plt.ylabel('mlogloss')

#plt.xscale('log')
plt.grid()

plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plots/{}/3class/eta{}_subsample{}_depth{}_round{}/mlogloss_v5.pdf'.format(year,eta,subsample,max_depth,num_round))




plt.clf()
Score = bst.get_fscore()
print ('Score is ', Score)
keys = np.array((Score.keys()))
values = np.array((Score.values()))

plt.barh(columns, values/float(max(values)))
plt.xlabel("Norm. feature importance")
#xgb.plot_importance(bst)
plt.tight_layout()
#plt.show()
plt.savefig('plots/{}/3class/eta{}_subsample{}_depth{}_round{}/fscore_v5.pdf'.format(year,eta,subsample,max_depth,num_round))


y_train = bst.predict(xg_train)[:,1]
y_test = bst.predict(xg_test)[:,1]
label_train = xg_train.get_label()
label_test = xg_test.get_label()
weight_train = xg_train.get_weight()
weight_test = xg_test.get_weight()

print ("label_train ", label_train)
print ("type ", type(label_train[0]))

datapostrain = np.zeros((label_train==1).sum())
datanegtrain = np.zeros((label_train==2).sum())
MCtrain = np.zeros((label_train==0).sum())

dataposwtrain = np.zeros((label_train==1).sum())
datanegwtrain = np.zeros((label_train==2).sum())
MCwtrain = np.zeros((label_train==0).sum())

datapostest = np.zeros((label_test==1).sum())
datanegtest = np.zeros((label_test==2).sum())
MCtest = np.zeros((label_test==0).sum())

dataposwtest = np.zeros((label_test==1).sum())
datanegwtest = np.zeros((label_test==2).sum())
MCwtest = np.zeros((label_test==0).sum())
ndatapos=0
ndataneg=0
nMC=0

for i in range(0,len(y_train)):
    if label_train[i]==0:
        MCtrain[nMC]=y_train[i]
        MCwtrain[nMC]=weight_train[i]
        nMC=nMC+1
    elif label_train[i]==1:
        datapostrain[ndatapos]=y_train[i]
        dataposwtrain[ndatapos]=weight_train[i]
        ndatapos=ndatapos+1
    else:
        datanegtrain[ndataneg]=y_train[i]
        datanegwtrain[ndataneg]=weight_train[i]
        ndataneg=ndataneg+1
    


ndatapos=0
ndataneg=0
nMC=0
for i in range(0,len(y_test)):
    if label_test[i]==0:
        MCtest[nMC]=y_test[i]
        MCwtest[nMC]=weight_test[i]
        nMC=nMC+1
    elif label_test[i]==1:
        datapostest[ndatapos]=y_test[i]
        dataposwtest[ndatapos]=weight_test[i]
        ndatapos=ndatapos+1
    else:
        datanegtest[ndataneg]=y_test[i]
        datanegwtest[ndataneg]=weight_test[i]
        ndataneg=ndataneg+1

'''print ("y_train", y_train)
print ("label_train", label_train)
print ("weight_train", weight_train)
print ("MCtrain", MCtrain)
print ("datatrain", datatrain)
print ("MCwtrain",MCwtrain)
print ("datawtrain",datawtrain)
print ("")

print ("y_test", y_test)
print ("label_test", label_test)
print ("weight_test", weight_test)
print ("MCtest", MCtest)
print ("datatest", datatest)
print ("MCwtest",MCwtest)
print ("datawtest",datawtest)'''
plt.figure(figsize=(12.8,9.6))
plt.clf()

low  = 0
high = 1
low_high = (low,high)
bins = 50

#################################################
plt.hist(
    datapostrain,
    color='r', 
    alpha=0.5, 
    range=low_high, 
    bins=bins,
    histtype='stepfilled', 
    density=True,
    weights=dataposwtrain,
    log=False,
    label='Datapos (train)'
)

#################################################
plt.hist(
    datanegtrain,
    color='g', 
    alpha=0.5, 
    range=low_high, 
    bins=bins,
    histtype='stepfilled', 
    density=True,
    weights=datanegwtrain,
    log=False,
    label='Dataneg (train)'
)

#################################################
plt.hist(
    MCtrain,
    color='b', 
    alpha=0.5, 
    range=low_high, 
    bins=bins,
    histtype='stepfilled', 
    density=True,
    weights=MCwtrain,    
    log=False,
    label='MC (train)'
)

#################################################
hist, bins = np.histogram(
    datapostest,
    bins=bins, 
    range=low_high, 
    density=True,
    weights=dataposwtest
)

width  = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
scale  = len(datapostest) / sum(hist)
err    = np.sqrt(hist * scale) / scale

plt.errorbar(
    center, 
    hist, 
    yerr=err, 
    fmt='o', 
    c='r', 
    label='Datapos (test)'
)

#################################################
hist, bins = np.histogram(
    datanegtest,
    bins=bins, 
    range=low_high, 
    density=True,
    weights=datanegwtest
)

width  = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
scale  = len(datanegtest) / sum(hist)
err    = np.sqrt(hist * scale) / scale

plt.errorbar(
    center, 
    hist, 
    yerr=err, 
    fmt='o', 
    c='g', 
    label='Dataneg (test)'
)

#################################################
hist, bins = np.histogram(
    MCtest,
    bins=bins, 
    range=low_high, 
    density=True,
    weights=MCwtest
)

width  = (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
scale  = len(MCtest) / sum(hist)
err    = np.sqrt(hist * scale) / scale

plt.errorbar(
    center, 
    hist, 
    yerr=err, 
    fmt='o', 
    c='b', 
    label='MC (test)'
)

#################################################
plt.xlabel('BDT output')
plt.ylabel('Arbitrary units')
plt.ylim(bottom=0)
plt.legend(loc='best',fontsize='large')
ks_datapos = ks_2samp(datapostrain, datapostest)
ks_dataneg = ks_2samp(datanegtrain, datanegtest)
ks_MC = ks_2samp(MCtrain, MCtest)
plt.suptitle('KS p-value: datpos = %.3f%s - datneg = %.3f%s - MC = %.2f%s' %(ks_datapos.pvalue * 100., '%', ks_dataneg.pvalue * 100., '%',ks_MC.pvalue * 100., '%'),fontsize='large')

# plt.tight_layout()
plt.savefig('plots/{}/3class/eta{}_subsample{}_depth{}_round{}/overtrain_v5.pdf'.format(year,eta,subsample,max_depth,num_round))



'''print ("y_train is ", y_train, " shape is ", y_train.shape)
print ("label_train is ", label_train, " shape is ", label_train.shape)
fpr_train, tpr_train, thresholds_train = roc_curve(label_train, y_train)
auc_train = roc_auc_score(label_train, y_train)

print ("fpr_train = ", fpr_train)
train_v = np.zeros(9)
for i in range (1,10):
    train_v[i-1]=np.interp(i*0.1,fpr_train,tpr_train)

print ("train_v, ", train_v)

fpr_test, tpr_test, thresholds_test = roc_curve(label_test, y_test)
auc_test = roc_auc_score(label_test, y_test)

print ("fpr_test = ", fpr_test)
test_v = np.zeros(9)
for i in range (1,10):
    test_v[i-1]=np.interp(i*0.1,fpr_test,tpr_test)

print ("test_v, ", test_v)

plt.clf()
plt.figure().set_size_inches(10.5, 9.5)
#plt.xscale('log')
plt.plot(fpr_train,tpr_train,lw=2, color='darkred',label='ROCtrain (AUC = %0.2f)' % auc_train)
plt.plot(fpr_test,tpr_test,lw=2, color='darkorange',label='ROCtest (AUC = %0.2f)' % auc_test)
plt.plot([0, 1], [0, 1], color='navy',label='random guessing',lw=2,linestyle='--')
plt.xlim([0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC curve',fontsize=25)                                                                                                  
plt.legend(loc="lower right",fontsize='xx-large')
plt.savefig('plots/{}/3class/eta{}_subsample{}_depth{}_round{}/ROC_v5.pdf'.format(year,eta,subsample,max_depth,num_round))

'''
Save_Dir = './model/{}/3class/{}_XGBV5_eta{}_subsample{}_depth{}_round{}.json'.format(year,year,eta,subsample,max_depth,num_round)
print(('Save the trained XGBoost model in {0}').format(Save_Dir))
bst.save_model(Save_Dir)

time_end=timer.time()
print('totally cost',time_end-time_start)


