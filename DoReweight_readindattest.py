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


from __future__ import division
from turtle import color
#get_ipython().run_line_magic('pylab', 'inline')
#figsize(16, 8)
import xgboost as xgb 
from xgboost import plot_importance
from xgboost import plot_tree
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import metrics
import pandas as pd
import root_numpy
from array import array
import ROOT as r
from ROOT import TCut
import time as timer
import math
import os

r.gROOT.SetBatch(True)

time_start=timer.time()
year = 2017
number = 100000
q2Bin = 4
parity = 0

# In[2]:
XGB_params_map = {}
Chi2_Pvalues_map_preMCW = {}
Chi2_Pvalues_map_afMCW = {}
KS_Pvalues_map_preMCW = {}
KS_Pvalues_map_afMCW = {}
Chi2_Pvalues_map_preMCW_LIST = {}
KS_Pvalues_map_preMCW_LIST = {}
Chi2_Pvalues_map_afMCW_LIST = {}
KS_Pvalues_map_afMCW_LIST = {}

'''Chi2_Pvalues_temps_preMCW = []
Chi2_Pvalues_temps_afMCW = []
KS_Pvalues_temps_preMCW = []
KS_Pvalues_temps_afMCW = []'''

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
variable("bLBSE"," bLBSE ",[100, 0, 0.04]) #significance
variable("bLBSsig"," bLBSsig ",[100, 0, 200])

variable("bCosAlphaBS"," bCosAlphaBS ",[100, 0.994, 1.0])

variable("bDCABS","bDCABS",[100, -0.015, 0.015])
variable("bDCABSE","bDCABSE",[100, 0, 0.01]) #significance
variable("bDCABSsig","bDCABSsig",[100, -5, 5])

variable("kstTrk1DCABS","kstTrk1DCABS",[100, -1, 1])
variable("kstTrk1DCABSE"," kstTrk1DCABSE",[100, 0, 0.5]) #significance
variable("kstTrk1DCABSsig"," kstTrk1DCABSsig",[100, -50,50])

variable("kstTrk2DCABS","kstTrk2DCABS",[100, -1, 1])
variable("kstTrk2DCABSE"," kstTrk2DCABSE",[100, 0, 0.5]) #significance
variable("kstTrk2DCABSsig"," kstTrk2DCABSsig",[100, -50,50])

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

variable("mumuLBS", "mumuLBS", [100,0,6.4])
variable("mumuLBSE", "mumuLBSE", [100,0,0.12])

variable("mumuDCA", "mumuDCA", [100,0,1])

variable("mumuCosAlphaBS", "mumuCosAlphaBS", [100,0.93,1])

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


#trkdca significance
#bvtxcl, blbssig, bcosalphaBS, 

###variables in note

variable("kstVtxCL","kstVtxCL",[100,0,1])


variable("weight","PUweight",[100,-1,3])


# In[3]:


def get_needed_branches_from_var(tree, var): #tree must be a TTree, cut must be a string or TCut
    if isinstance(var, TCut): var = var.GetTitle()
    list_of_branches = [x.GetName() for x in tree.GetListOfBranches()]
    list_of_needed_branches = [x for x in list_of_branches if x in var]
    return list_of_needed_branches

def only_activate_variables(tree, variables):
    tree.SetBranchStatus("*", 0)
    for variable in variables:
        branches = get_needed_branches_from_var(tree, variable)
        for branch in branches:
            tree.SetBranchStatus(branch, 1)

def plot_var(KS_Pvalues_temps_preMCW,Chi2_Pvalues_temps_preMCW,KS_Pvalues_temps_afMCW,Chi2_Pvalues_temps_afMCW,varname, cut="1", reweight=False, plotpath="plottest/"):
    only_activate_variables(rdata, [varname, cut]) #this speeds things up
    only_activate_variables(MC, ["weight", "tagged_mass", varname, cut]) #this speeds things up
    rdata.SetBranchStatus("nsigb4p1_{}_sw".format(year), 1) #this ensure the introduction of SWeight of real data for signal extraction from background
    if reweight: 
        MC.SetBranchStatus("MCw", 1) #this needs to be done manually because it's on the friend tree
    
    hdata = r.TH1F("hdata", "hdata", variables[varname].get_nbins(), variables[varname].get_xmin(), variables[varname].get_xmax())
    swname = "nsigb4p1_{}_sw*".format(year)
    #rdata.Draw(varname+">>hdata", cut, "goff")
    #rdata.Draw(varname+">>hdata", swname+"{0}".format(cut), "goff")
    rdata.Draw(varname+">>hdata", swname+"{0}".format(cut), "goff", 100000, 0)
    if hdata.Integral()==0:
        print("Empty histogram!")
        return
    hdata.Scale(1./hdata.Integral())
    
    hMC = r.TH1F("hMC", "hMC", hdata.GetNbinsX(), hdata.GetXaxis().GetXmin(), hdata.GetXaxis().GetXmax())
    #wname = "MCw*" if reweight else ""
    #MC.Draw(varname+">>hMC", "PUweight*"+wname+"({0})".format(cut), "same goff")
    #MC.Draw(varname+">>hMC", wname+"({0})".format(cut), "same goff")
    wname = ""
    MC.Draw(varname+">>hMC", "weight*"+wname+"({0})".format(cut), "same goff", 100000, 0)
    if hMC.Integral()==0:
        print("Empty histogram!")
        return
    hMC.Scale(1./hMC.Integral())
    hMC.SetLineColor(r.kRed)
    hMC.SetFillColor(r.kRed)
    hMC.SetFillStyle(3244)
    Chi2_Pvalues_temps_preMCW.append(hMC.Chi2Test(hdata,"WWP"))
    KS_Pvalues_temps_preMCW.append(hMC.KolmogorovTest(hdata,""))
    print("========================================================")
    print("Chi2 test of {} (MC vs data): P-value = {}".format(varname,Chi2_Pvalues_temps_preMCW[-1]))
    print("KS test of {} (MC vs data): P-value = {}".format(varname,KS_Pvalues_temps_preMCW[-1]))
    print("--------------------------------------------------------")
    #del hMC
    hWMC = r.TH1F("hWMC", "hWMC", hdata.GetNbinsX(), hdata.GetXaxis().GetXmin(), hdata.GetXaxis().GetXmax())
    wname = "MCw*"
    MC.Draw(varname+">>hWMC", "weight*"+wname+"({0})".format(cut), "same goff", 100000, 0)
    if hWMC.Integral()==0:
        print("Empty histogram!")
        return
    hWMC.Scale(1./hWMC.Integral())
    hWMC.SetLineColor(r.kRed)
    hWMC.SetFillColor(r.kRed)
    hWMC.SetFillStyle(3244)  
    Chi2_Pvalues_temps_afMCW.append(hWMC.Chi2Test(hdata,"WWP"))  
    KS_Pvalues_temps_afMCW.append(hWMC.KolmogorovTest(hdata,""))  
    print("========================================================")
    print("Chi2 test of weighted {} (MC vs data): P-value = {}".format(varname,Chi2_Pvalues_temps_afMCW[-1]))
    print("KS test of weighted {} (MC vs data): P-value = {}".format(varname,KS_Pvalues_temps_afMCW[-1]))
    print("--------------------------------------------------------\n\n\n")
    
    c = r.TCanvas("canvas","canvas",1600,600)
    c.Divide(2,1)
    
    ############################################################################
    c.cd(1)
    hs = r.THStack()
    hs.Add(hMC, "hist")
    hdata.SetMarkerStyle(9)
    hs.Add(hdata, "P")
    #Upper plot will be in pad1
    pad1 = r.TPad("pad1", "pad1", 0, 0.25, 1, 1.0)
    pad1.SetTopMargin(0.05)
    pad1.SetBottomMargin(0.05)
    pad1.SetGridx()         #Vertical grid
    pad1.Draw()             #Draw the upper pad: pad1
    pad1.cd()               #pad1 becomes the current pad
    hs.Draw("nostack")
    hs.SetTitle("")
    #Y axis h1 plot settings
    hs.GetYaxis().SetTitleSize(25)
    hs.GetYaxis().SetTitleFont(43)
    hs.GetYaxis().SetTitleOffset(0.5)
    hs.GetYaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    hs.GetYaxis().SetLabelSize(20)
    hs.GetXaxis().SetTitleSize(0)
    hs.GetXaxis().SetLabelSize(15)
    
    c.cd(1)          #Go back to the main canvas before defining pad2
    pad2 = r.TPad("pad2", "pad2", 0, 0.0, 1, 0.25)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.5)
    pad2.SetGridx() #vertical grid
    pad2.Draw()
    pad2.cd()       #pad2 becomes the current pad
    
    #Define the ratio plot
    hratio = hdata.Clone("hratio")
    hratio.SetLineColor(r.kBlack)
    hratio.SetMinimum(0.35)  #Define Y
    hratio.SetMaximum(1.65) #range
    hratio.GetXaxis().SetTitle(variables[varname].title)
    #hratio.Sumw2()
    hratio.SetStats(0)      #No statistics on lower plot
    hratio.Divide(hMC)
    hratio.Draw("ep")
    hratio.SetTitle("")
    
    #Y axis ratio plot settings
    hratio.GetYaxis().SetTitle("ratio")
    hratio.GetYaxis().SetNdivisions(505)
    hratio.GetYaxis().SetTitleSize(20)
    hratio.GetYaxis().SetTitleFont(43)
    hratio.GetYaxis().SetTitleOffset(1.0)
    hratio.GetYaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    hratio.GetYaxis().SetLabelSize(20)
    
    #X axis ratio plot settings
    hratio.GetXaxis().SetTitleSize(20)
    hratio.GetXaxis().SetTitleFont(43)
    hratio.GetXaxis().SetTitleOffset(4.8)
    hratio.GetXaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    hratio.GetXaxis().SetLabelSize(15)
    hratio.GetXaxis().SetLabelOffset(0.01)
    
    line = r.TLine(hratio.GetXaxis().GetXmin(),1,hratio.GetXaxis().GetXmax(), 1)
    line.SetLineStyle(3)
    line.Draw()
    
    ############################################################################
    c.cd(2)
    whs = r.THStack()
    whs.Add(hWMC, "hist")
    hdata.SetMarkerStyle(9)
    whs.Add(hdata, "P")
    #Upper plot will be in pad1
    wpad1 = r.TPad("pad1", "pad1", 0, 0.25, 1, 1.0)
    wpad1.SetTopMargin(0.05)
    wpad1.SetBottomMargin(0.05)
    wpad1.SetGridx()         #Vertical grid
    wpad1.Draw()             #Draw the upper pad: pad1
    wpad1.cd()               #pad1 becomes the current pad
    whs.Draw("nostack")
    whs.SetTitle("")
    #Y axis h1 plot settings
    whs.GetYaxis().SetTitleSize(25)
    whs.GetYaxis().SetTitleFont(43)
    whs.GetYaxis().SetTitleOffset(0.5)
    whs.GetYaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    whs.GetYaxis().SetLabelSize(20)
    whs.GetXaxis().SetTitleSize(0)
    whs.GetXaxis().SetLabelSize(15)
    
    c.cd(2)          #Go back to the main canvas before defining pad2
    wpad2 = r.TPad("pad2", "pad2", 0, 0.0, 1, 0.25)
    wpad2.SetTopMargin(0)
    wpad2.SetBottomMargin(0.5)
    wpad2.SetGridx() #vertical grid
    wpad2.Draw()
    wpad2.cd()       #pad2 becomes the current pad
    
    #Define the ratio plot
    whratio = hdata.Clone("hratio_W")
    whratio.SetLineColor(r.kBlack)
    whratio.SetMinimum(0.35)  #Define Y
    whratio.SetMaximum(1.65) #range
    whratio.GetXaxis().SetTitle(variables[varname].title)
    #whratio.Sumw2()
    whratio.SetStats(0)      #No statistics on lower plot
    whratio.Divide(hWMC)
    whratio.Draw("ep")
    whratio.SetTitle("")
    
    #Y axis ratio plot settings
    whratio.GetYaxis().SetTitle("ratio")
    whratio.GetYaxis().SetNdivisions(505)
    whratio.GetYaxis().SetTitleSize(20)
    whratio.GetYaxis().SetTitleFont(43)
    whratio.GetYaxis().SetTitleOffset(1.0)
    whratio.GetYaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    whratio.GetYaxis().SetLabelSize(20)
    
    #X axis ratio plot settings
    whratio.GetXaxis().SetTitleSize(20)
    whratio.GetXaxis().SetTitleFont(43)
    whratio.GetXaxis().SetTitleOffset(4.8)
    whratio.GetXaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    whratio.GetXaxis().SetLabelSize(15)
    whratio.GetXaxis().SetLabelOffset(0.01)
    
    wline = r.TLine(hratio.GetXaxis().GetXmin(),1,hratio.GetXaxis().GetXmax(), 1)
    wline.SetLineStyle(3)
    wline.Draw()

    ############################################################################
    #c.SaveAs(plotpath+wname+varname+".png")
    #c.SaveAs(plotpath+wname+varname+".pdf")
    c.SaveAs(plotpath+varname+".png")
    
    #at the moment I'm saving the legend in a separate canvas EACH TIME. This is a quick workaraound and it is inefficient. It needs to be fixed later.
    cleg = r.TCanvas("legend","legend",500,350)
    leg = r.TLegend(0.1,0.1,0.9,0.9)
    leg.AddEntry(hdata, "data", "ep")
    leg.AddEntry(hMC, "simulation", "f")
    leg.Draw()
    cleg.SaveAs(plotpath+"legend"+".png")
    #cleg.SaveAs(plotpath+"legend"+".pdf")
    
    #cleanup
    hs.Delete()
    #hratio.Delete()
    hMC.Delete()
    hdata.Delete()
    pad1.Delete()
    pad2.Delete()


# ## Prepare for data
#     

# In[4]:

print("Start------------------------------------------")

#This column is used to draw picture
columns_draw = [
                #'MumEta','MumPt',
                #'MupEta','MupPt',
                #'MuMuLxySigmaBs','MuMuDca','MuMuCosAlphaBs',
                #'TrkEta','TrkPt','Trkdcasigbs',
                #'Bvtxcl','Blxysig','Bcosalphabs2D','Bcosalphabs',
                #'BEta','BPt','Bmass',
                #'CosThetaL'
                ]

# In[5]:


#Draw original distribution before reweighting


print("TTrees preparation------------------------------------------")

#cut_bpeak = "(Bmass>5.0 && Bmass<5.5)"
cut_bpeak = "(tagged_mass>5.0 && tagged_mass<5.6)"


rdata = r.TChain("ntuple")
#rdata.Add('/eos/user/c/cjiang/selected/data/2018/plainized_data_cutJpsi0_all_2018_UL_MINIAODv1_all_aftercutJpsi0.root')
rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b{}_{}_p{}.root'.format(q2Bin,year,parity))

MC = r.TChain("ntuple")
MC.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoMCDataset_b{}_{}_p{}.root'.format(q2Bin,year,parity))

MC_friend = r.TTree("wTree", "weights tree")
leafValues = array("f", [0.0])
weight_branch = MC_friend.Branch("MCw", leafValues,"MCw[1]/F")
print("Wtree builded")

#for v in variables.keys(): plot_var(v, cut_bpeak, False)

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# In[6]:

print("Training samples preparation------------------------------------------")

#This column is used as input to the XGBoost, branchs in your tree

#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','MumEta','Bcosalphabs','Mumdcabserr']
#columns = ['Trkdcabserr','Blsbs','Blsbserr','Mumdcabserr','Mupdcabserr','MumEta']
#columns = ['Trkdcabserr','Blsbs','Blsbserr','Mumdcabserr','Mupdcabserr','MumEta','Bcosalphabs']
#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','MumEta','Bcosalphabs']
#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','MumEta','Mumdcabserr','Mupdcabserr','Trkdcabserr','Blsbserr']
#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','MumEta','MumPhi','Mumdcabserr','Mupdcabserr','Mumulsbserr']
#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','MumEta','MumPhi','MupPhi','Mumdcabserr','Mupdcabserr','Mumulsbserr']
#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','MumEta','MumPhi','MupPhi','MumPt','MupPt']
#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','Bcosalphabs','MumEta','MumPhi','MumudR']
#columns = ['Trkdcabs','Trkdcabserr','Bvtxcl','Blsbs','Blsbserr','TrkPt','Bcosalphabs']
#columns = ['Trkdcabs','Bvtxcl','Blsbs','TrkPt','Bcosalphabs','MumEta','MumPhi','MumPt','MupPt','BPt','Mumdcabserr','Mupdcabserr','Trkdcabserr','Blsbserr']

#columns = ['Trkdcasigbs','Bvtxcl','Blxysig','TrkPt','Bcosalphabs','MumEta','MumPhi']
columns = ['bVtxCL', 'bLBSsig', 'bCosAlphaBS', 'bDCABSsig', 'kstTrk1DCABSsig', 'kstTrk2DCABSsig','kstTrk1MinIP2D','kstTrk2MinIP2D','sum_isopt_04','bPt','bEta','bPhi','kstTrk1Pt','kstTrk2Pt','mumPt','mupPt']
#columns = ['bVtxCL', 'bLBSsig', 'bCosAlphaBS', 'bDCABSsig', 'kstTrkmDCABSsig', 'kstTrkpDCABSsig','kstTrkmMinIP2D','kstTrkpMinIP2D','sum_isopt_04','bPt','bEta','bPhi']
sw_branch = ['nsigb4p1_{}_sw'.format(year)]

#Read data using root_numpy, define the tree name, branch list

#data_ori = root_numpy.root2array('/eos/user/c/cjiang/selected/data/2018/plainized_data_cutJpsi0_all_2018_UL_MINIAODv1_all_aftercutJpsi0.root',treename='tree', branches=columns, start=0, stop=100000)
data_ori = root_numpy.tree2array(rdata,branches=columns,start=0, stop=number)
print("Data sample readed------------------------------------------")
#phsp_ori = root_numpy.root2array('/eos/user/c/cjiang/selected/MC/2018/plainized_PileupRweight_BuToMuMuK_SIM_2018_UL_MINIAODv1_all_aftercutJpsi0.root', treename='tree',branches=columns, start=0, stop=100000)
phsp_ori = root_numpy.tree2array(MC,branches=columns,start=0,stop=number)
print("MC sample readed------------------------------------------")
# JpsiK dataset's SWeights to extract signal peak to compare with MC distribution
JpsiKSignal_SW = root_numpy.tree2array(rdata,branches=sw_branch,start=0,stop=number)
print("Sweights readed------------------------------------------")

#Translate data into pandas formation
data_only_X=pd.DataFrame(data_ori,columns=columns)
phsp_only_X=pd.DataFrame(phsp_ori,columns=columns)
sw_sig_RD_X=pd.DataFrame(JpsiKSignal_SW)
data_signal_sumEntries = (sw_sig_RD_X.sum())[0]
print("data_signal_sumEntries = {0}".format((sw_sig_RD_X.sum())[0]))
sf_phsp_vs_data=(len(phsp_only_X))/(data_signal_sumEntries)

#Make labels for each data, data marks as 1, MC marks as 0 
data_only_Y=np.ones(len(data_only_X))
phsp_only_Y=np.zeros(len(phsp_only_X))

data=pd.DataFrame(data_ori)
phsp=pd.DataFrame(phsp_ori)
sw_sig_RD=np.ones(len(phsp_ori))

#for MC truth information

# In[7]:

#Tanslate into array format
data_only_a=np.array(data)
phsp_only_a=np.array(phsp)
sw_sig_RD_a=np.array(sw_sig_RD_X)
sw_sig_RD_a=sw_sig_RD_a.reshape(-1,1)

if((data_only_a.shape)[0] > (sw_sig_RD_a.shape)[0]): 
    sw_sig_RD_a=np.concatenate([sw_sig_RD_a, (np.ones((len(data_only_a) - len(sw_sig_RD_a),1)).reshape(-1,1))], axis=0)
    #sw_sig_RD_a=np.concatenate([sw_sig_RD_a, [1] * ((data_only_a.shape)[0] - (sw_sig_RD_a.shape)[0])], axis=0)
if((data_only_a.shape)[0] < (sw_sig_RD_a.shape)[0]): 
    data_only_a=np.concatenate([data_only_a, (np.zeros((len(sw_sig_RD_a) - len(data_only_a),1)).reshape(-1,1))], axis=0)
    #data_only_a=np.concatenate([data_only_a, [0] * ((sw_sig_RD_a.shape)[0] - (data_only_a.shape)[0])], axis=0)
#sf_phsp_vs_data=((phsp_only_a.shape())[0])/((data_only_a.shape())[0])


#Shuffle the data and split the data
from sklearn.model_selection import train_test_split
data_all = np.concatenate([data_only_a, phsp_only_a], axis=0)
#print("data_all")
#print(data_all)
ones_phsp = (np.array([1]*len(phsp_only_a))).reshape(-1,1)
sw_sig_RD_a = sf_phsp_vs_data * sw_sig_RD_a
sweights_all = np.concatenate([sw_sig_RD_a, ones_phsp], axis=0) 
#for k in range(0,len(sweights_all)): sweights_all[k][0] = 1
#print("sweights_all")
#print(sweights_all)
data_all = np.concatenate([data_all, sweights_all], axis=1)
#print("new_data_all")
#print(data_all)
labels_all = np.array([1] * len(data_only_a) + [0] * len(phsp_only_a))
#print("new_labels_all")
#print(labels_all)
train_X,test_X,train_Y,test_Y=train_test_split(data_all,labels_all,test_size=0.1,stratify=labels_all)


# In[8]:
# ## Initial the xgboost

sw_train = train_X[:,-1]
#sw_train = sw_train.reshape(1,-1)
sw_train = sw_train.tolist()
#print("sw_train")
#print(sw_train)
sw_test = test_X[:,-1]
#sw_test = sw_test.reshape(1,-1)
sw_test = sw_test.tolist()
#print("sw_test")
#print(sw_test)
train_X = train_X[:,:-1]
#print("train_X")
#print(train_X)
test_X = test_X[:,:-1]
#print("test_X")
#print(test_X)

#Prepare the input data for XGBoost
print("xg_train")
xg_train = xgb.DMatrix(train_X, label=train_Y, weight=sw_train)
print("xg_test")
xg_test = xgb.DMatrix(test_X, label=test_Y, weight=sw_test)
print("xg_data_only")
xg_data_only = xgb.DMatrix(data_only_X, label=data_only_Y, weight=(sw_sig_RD_a))
print("xg_phsp_only")
xg_phsp_only = xgb.DMatrix(phsp_only_X, label=phsp_only_Y)

'''
params={
'booster':'gbtree',
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.007, # 如同学习率
'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':6, # 构建树的深度，越大越容易过拟合
'gamma':0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样 
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
#'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
#'objective': 'multi:softmax', #多分类的问题
'num_class':2, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子
#'eval_metric': 'auc'
}
'''







for itr in range(0,125):
    # setup parameters for xgboost
    params = {}
    # use softmax multi-class classification
    params['objective'] = 'multi:softmax'
    # scale weight of positive examples
    params['eta'] = 0.05+0.05*((itr//1)%5)
    params['max_depth'] = 3+1*((itr//5)%5)
    params['silent'] = 1
    params['nthread'] = 4
    params['num_class'] = 2
    
    # do the same thing again, but output probabilities
    params['objective'] = 'multi:softprob'
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 100+50*((itr//25)%5) # 迭代次数
    params['num_round'] = num_round
    print ("itr is", itr, "params ", params)
    print("training==================================")
    bst = xgb.train(params, xg_train, num_round, watchlist)
    
    #save params in dictionary -- XGB_params_map
    #params['num_round'] = num_round
    XGB_params_map[itr] = params
    params = {}
    
    # In[9]:
    
    print("Machine learning------------------------------------------")
    
    #Calculate the error from the XGBoost, this number could be large because the data has large number of phsp component. XGBoost can not distinguish them correctly.
    pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 2)
    pred_label = np.argmax(pred_prob, axis=1)
    
    error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
    print('Test error using softprob = {}'.format(error_rate))
    
    Save_Dir = './validate/trained_model{}.json'.format(itr)
    #print(('Save the trained XGBoost model in {0}').format(Save_Dir))
    #bst.save_model(Save_Dir)
    
    # In[10]:
    
    
    #Calculate weight for each event using probability
    #pr_test=np.array(bst.predict(xg_test,validate_features=False).reshape(test_Y.shape[0], 2))
    #pr_phsp=np.array(bst.predict(xg_phsp_only,validate_features=False).reshape(phsp_only_Y.shape[0], 2))
    
    #trained_bst = xgb.Booster()
    #trained_bst = trained_bst.load_model(Save_Dir)
    
    trained_bst = bst
    #trained_bst = xgb.Booster(model_file=Save_Dir)
    pr_phsp=np.array(trained_bst.predict(xg_phsp_only,validate_features=False).reshape(phsp_only_Y.shape[0], 2))
    
    #weight_test=pr_test[:,1]/pr_test[:,0]
    weight_phsp=pr_phsp[:,1]/pr_phsp[:,0]
    print("MC_weights: weight_phsp=pr_phsp")
    #print(weight_phsp)
    
    # In[11]:
    
    print("MC weights------------------------------------------")
    
    #MCwFile = r.TFile("./JPsiK_data_MC_weights.root","RECREATE")
    
    MC_friend = r.TTree("wTree", "weights tree") # to reset weights tree
    leafValues = array("f", [0.0])
    weight_branch = MC_friend.Branch("MCw", leafValues,"MCw[1]/F")
    for val in weight_phsp:
        leafValues[0] = val
        MC_friend.Fill()
    
    #MCwFile.cd()
    #MC_friend.Write()
    
    MC.AddFriend(MC_friend)
    
    Chi2_Pvalues_temps_preMCW = []
    KS_Pvalues_temps_preMCW = []
    Chi2_Pvalues_temps_afMCW = []
    KS_Pvalues_temps_afMCW = []
    #for v in variables.keys(): plot_var(v, cut_bpeak, True)
    plotpath_temp = (("plottest/params_{0}/").format(itr))
    if not (os.path.exists(plotpath_temp)): os.makedirs(plotpath_temp)
    for v in columns: plot_var(KS_Pvalues_temps_preMCW,Chi2_Pvalues_temps_preMCW,KS_Pvalues_temps_afMCW,Chi2_Pvalues_temps_afMCW,v, cut_bpeak, True, plotpath_temp)
    Chi2_Pvalues_map_preMCW_LIST[itr] = Chi2_Pvalues_temps_preMCW
    KS_Pvalues_map_preMCW_LIST[itr] = KS_Pvalues_temps_preMCW
    Chi2_Pvalues_map_afMCW_LIST[itr] = Chi2_Pvalues_temps_afMCW
    KS_Pvalues_map_afMCW_LIST[itr] = KS_Pvalues_temps_afMCW
    
    Chi2_Pvalues_product_temp_preMCW = 1
    KS_Pvalues_product_temp_preMCW = 1
    for chi in Chi2_Pvalues_temps_preMCW: Chi2_Pvalues_product_temp_preMCW+=( (math.log10(chi) if chi!=0 else 0) ) # in case of P-value is too small, so I times 10
    for ks in KS_Pvalues_temps_preMCW: KS_Pvalues_product_temp_preMCW+=( (math.log10(ks) if ks!=0 else 0) ) # in case of P-value is too small, so I times 10
    Chi2_Pvalues_map_preMCW[itr] = Chi2_Pvalues_product_temp_preMCW
    KS_Pvalues_map_preMCW[itr] = KS_Pvalues_product_temp_preMCW
    
    Chi2_Pvalues_product_temp_afMCW = 1
    KS_Pvalues_product_temp_afMCW = 1
    for chi in Chi2_Pvalues_temps_afMCW: Chi2_Pvalues_product_temp_afMCW+=( (math.log10(chi) if chi!=0 else 0) ) # in case of P-value is too small, so I times 10
    for ks in KS_Pvalues_temps_afMCW: KS_Pvalues_product_temp_afMCW+=( (math.log10(ks) if ks!=0 else 0) ) # in case of P-value is too small, so I times 10
    Chi2_Pvalues_map_afMCW[itr] = Chi2_Pvalues_product_temp_afMCW
    KS_Pvalues_map_afMCW[itr] = KS_Pvalues_product_temp_afMCW
    
    #MCwFile.Close()


print("\n\n\n==================================== Summary ==========================================")

print("XGB_params_map"); print XGB_params_map
print("---------------------------------------------------\n")

print("PreMCW Chi2_Pvalues: "); print Chi2_Pvalues_map_preMCW
print("---------------------------------------------------\n")
print("PreMCW KS_Pvalues: "); print KS_Pvalues_map_preMCW
print("---------------------------------------------------\n")

print("AfterMCW Chi2_Pvalues: "); print Chi2_Pvalues_map_afMCW
print("---------------------------------------------------\n")
print("AfterMCW KS_Pvalues: "); print KS_Pvalues_map_afMCW
print("---------------------------------------------------\n")


print("Chi2_Pvalues_map_preMCW_LIST")
print(Chi2_Pvalues_map_preMCW_LIST)
print("---------------------------------------------------\n")
print("KS_Pvalues_map_preMCW_LIST")
print(KS_Pvalues_map_preMCW_LIST)
print("---------------------------------------------------\n")
print("Chi2_Pvalues_map_afMCW_LIST")
print(Chi2_Pvalues_map_afMCW_LIST)
print("---------------------------------------------------\n")
print("KS_Pvalues_map_afMCW_LIST")
print(KS_Pvalues_map_afMCW_LIST)
print("---------------------------------------------------\n")

max_Chi2P = 0
max_Chi2P_ind = 0
for k in Chi2_Pvalues_map_afMCW.keys():
    if Chi2_Pvalues_map_afMCW[k] > max_Chi2P:
        max_Chi2P = Chi2_Pvalues_map_afMCW[k]
        max_Chi2P_ind = k
print(("Best params set: index={}, max_Chi2P={}").format(max_Chi2P_ind,max_Chi2P))


max_KSP = 0
max_KSP_ind = 0
for k in KS_Pvalues_map_afMCW.keys():
    if KS_Pvalues_map_afMCW[k] > max_KSP:
        max_KSP = KS_Pvalues_map_afMCW[k]
        max_KSP_ind = k
print(("Best params set: index={}, max_KSP={}").format(max_KSP_ind,max_KSP))

print("=======================================================================================")




# In[12]:
time_end=timer.time()
print('totally cost',time_end-time_start)

