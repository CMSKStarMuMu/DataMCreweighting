from ROOT import TTree,TFile,TChain
import root_pandas
import numpy as np
import pandas as pd
import root_numpy
from array import array
pd.options.mode.chained_assignment = None  # default='warn'
import sys
import ROOT as r
q2Bin = int(sys.argv[1])
parity = int(sys.argv[2])
year = int(sys.argv[3])
mc_sigma = 0.040
mc_mass  = 5.27783 
JPsiMass_ = 3.096916
nSigma_psiRej = 3.

selBmass = '&& (bMass*tagB0 + (1-tagB0)*bBarMass)   > {M}-3*{S}   && \
            (bMass*tagB0 + (1-tagB0)*bBarMass)   < {M}+3*{S} '\
            .format(M=mc_mass,S=mc_sigma)

selData = '( pass_preselection ==1 ) && \
            (abs(mumuMass - {JPSIM}) < {CUT}*mumuMassE) &&  eventN%2=={Parity}'\
            .format( JPSIM=JPsiMass_, CUT=nSigma_psiRej, Parity=parity)

selMC = selData + ' && (trig==1)' 
selMCtruth = '&& truthMatchMum == 1 && truthMatchMup == 1 && truthMatchTrkm == 1 && truthMatchTrkp == 1'


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


#variable of BDT
variable("kstarmass", "kstar tagged mass", [100, 0.74, 1.05 ])
variable("bVtxCL","B decay vertex CL",[100, 0.0, 1.0])
variable("bLBS"," bLBS",[100, 0, 1.5])
variable("bLBSE"," bLBSE",[100, 0, 0.025]) #significance
#variable("bLBSsig_scaled"," bLBSsig_scaled ",[100, 0, 200])

variable("bCosAlphaBS","bCosAlphaBS",[100, 0.9996, 1.0])

variable("bDCABS","bDCABS",[100, -0.015, 0.015])
variable("bDCABSE","bDCABSE_scaled",[100, 0, 0.005]) #significance

variable("kstTrk1DCABS","kstTrk1DCABS",[100, -1, 1])
variable("kstTrk2DCABS","kstTrk2DCABS",[100, -1, 1])
variable("kstTrk1DCABSE","kstTrk1DCABSE",[100, 0, 0.025])
variable("kstTrk2DCABSE","kstTrk2DCABSE",[100, 0, 0.035])


variable("kstTrk2MinIP2D","kstTrk2MinIP2D ",[100, 0, 1])
variable("kstTrk1MinIP2D","kstTrk1MinIP2D ",[100, 0, 1])

variable("sum_isopt_04","sum_isopt_04 ",[100, 0, 40])

variable("mu1Pt", "mu1PT", [100,0,40])
variable("mu2Pt", "mu2PT", [100,0,40])

variable("mu1Eta", "mu1Eta", [100,-2.5,2.5])
variable("mu2Eta", "mu2Eta", [100,-2.5,2.5])

variable("mu1Phi", "mu1Phi", [100,-3.14159,3.14159])
variable("mu2Phi", "mu2Phi", [100,-3.14159,3.14159])

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



columns_draw = [
                "bPt","bEta","bVtxCL","bDCABS","bDCABSE","bLBS","bLBSE",'bCosAlphaBS',
                "mu1Pt","mu1Eta","mu1Phi",
                "mu2Pt","mu2Eta","mu2Phi",
                "kstTrk2Pt","kstTrk2Eta","kstTrk2Phi",
                "kstTrk1Pt","kstTrk1Eta","kstTrk1Phi",
                "cos_theta_l","cos_theta_k","phi_kst_mumu",
                "kstTrk1DCABS","kstTrk1DCABSE",
                "kstTrk2DCABS","kstTrk2DCABSE",
                "sum_isopt_04"

                ]

columns_error = ["bLBSE","bDCABSE","kstTrk1DCABSE","kstTrk2DCABSE"]

columns = ['bCosAlphaBS']

color = [r.kRed,r.kBlue]

rdata = TChain("ntuple")
rMC = []
rMC_ori = TChain("ntuple")
rMC_rw= TChain("ntuple")
rdata.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_data_beforsel.root".format(year,year))
rMC_ori.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_MC_JPSI_scale_and_preselection_XGBV5_p{}_new.root".format(year,year,parity))
rMC_rw.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_MC_JPSI_scale_and_preselection_XGBV5_p{}_new.root".format(year,year,parity))

rMC.append(rMC_ori)
rMC.append(rMC_rw)

label = ["", "XGBV5"]


def plot_var(varname):

    hdata = r.TH1F('h{}_data'.format(varname),'h{}_data'.format(varname),variables[varname].get_nbins(), variables[varname].get_xmin(), variables[varname].get_xmax())
    rdata.Draw('{}>>h{}_data'.format(varname,varname),'nsig_sw*({})'.format(selData),'goff')
    hdata.Scale(1./hdata.Integral())

    hMC=[]
    hratio=[]

    for i in range(0,2):
        hMC.append(r.TH1F('h{}_MC_{}'.format(varname,label[i]),'h{}_MC_{}'.format(varname,label[i]),variables[varname].get_nbins(), variables[varname].get_xmin(), variables[varname].get_xmax()))
        if i==0:
            rMC[i].Draw('{}>>h{}_MC_{}'.format(varname,varname,label[i]),'weight*DCratew*({})'.format(selMC+selMCtruth),'goff')
        else:
            rMC[i].Draw('{}>>h{}_MC_{}'.format(varname,varname,label[i]),'weight*DCratew*MCw*({})'.format(selMC+selMCtruth),'goff')
        hMC[i].Scale(1./hMC[i].Integral())
        print ("KStest of {} _ {} is \n {}".format(varname,label[i], hMC[i].KolmogorovTest(hdata,"D")))
        print ("Chi2 test of {} _ {} is \n {}".format(varname,label[i], hMC[i].Chi2Test(hdata,"WWP")))
        hratio.append(hMC[i].Clone('h{}_ratio_{}'.format(varname,label[i])))
        hratio[i].Divide(hdata)

    c = r.TCanvas("canvas","canvas",1600,1200)
    #pad1
    pad1_3In1 = r.TPad("pad1", "pad1", 0, 0.25, 1, 1.0)
    pad1_3In1.SetTopMargin(0.05)
    pad1_3In1.SetBottomMargin(0.05)
    pad1_3In1.SetGridx()         #Vertical grid
    pad1_3In1.Draw()             #Draw the upper pad: pad1
    pad1_3In1.cd()               #pad1 becomes the current pad

    hs = r.THStack()
    hdata.SetMarkerStyle(3)
    hdata.SetMarkerColor(r.kBlack)
    hs.Add(hdata, "P")
    for i in range(0,2):
        hMC[i].SetLineColor(color[i])
        hMC[i].SetLineWidth(3)
        hs.Add(hMC[i],"hist")
    #print('KS value of {} scaled of {} is {}'.format(label[i],variable,hMC.KolmogorovTest(hdata[i],"")))
    hs.Draw("nostack")
    hs.SetTitle("Comparison of bin4 {}".format(varname))

    #Y axis h1 plot settings
    hs.GetYaxis().SetTitleSize(30)
    hs.GetYaxis().SetTitleFont(43)
    hs.GetYaxis().SetTitleOffset(0.5)
    hs.GetYaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    hs.GetYaxis().SetLabelSize(25)
    hs.GetXaxis().SetTitleSize(0)
    hs.GetXaxis().SetLabelSize(20)

    if 'Eta' in v or 'Phi' in v or 'cos_theta_l' in v or 'phi' in v:
        leg = r.TLegend(0.35,0.1,0.55,0.35)
        leg.AddEntry(hdata, "data", "ep")
        for i in range(0,2):
            leg.AddEntry(hMC[i], "MC {}".format(label[i]), "lf")
        leg.Draw()

    elif v=='bCosAlphaBS':
        leg = r.TLegend(0.35,0.1,0.55,0.35)
        leg.AddEntry(hdata, "data", "ep")
        for i in range(0,2):
            leg.AddEntry(hMC[i], "MC {}".format(label[i]), "lf")
        leg.Draw()
    
    else:
        leg = r.TLegend(0.7,0.7,0.90,0.95)
        leg.AddEntry(hdata, "data", "ep")
        for i in range(0,2):
            leg.AddEntry(hMC[i], "MC {}".format(label[i]), "lf")
        leg.Draw()


    c.cd()
    pad2_3In1 = r.TPad("pad2", "pad2", 0, 0.0, 1, 0.25)
    pad2_3In1.SetTopMargin(0)
    pad2_3In1.SetBottomMargin(0.5)
    pad2_3In1.SetGridx() #vertical grid
    pad2_3In1.Draw()
    pad2_3In1.cd()       #pad2 becomes the current pad

    hratio_MCvsdata = r.THStack()
    for i in range(0,2):
        hratio[i].SetMarkerColor(color[i])
        hratio[i].SetLineColor(color[i])
        hratio[i].SetLineWidth(3)
        #hratio[i].SetMarkerStyle(2)
        hratio_MCvsdata.Add(hratio[i],"P")

    hratio_MCvsdata.Draw("nostack")
    hratio_MCvsdata.SetMinimum(0.35)  #Define Y
    hratio_MCvsdata.SetMaximum(1.65) #range
    hratio_MCvsdata.GetXaxis().SetTitle(varname)
    hratio_MCvsdata.SetTitle("")
    #Y axis ratio plot settings
    hratio_MCvsdata.GetYaxis().SetTitle("ratio")
    hratio_MCvsdata.GetYaxis().SetNdivisions(505)
    hratio_MCvsdata.GetYaxis().SetTitleSize(30)
    hratio_MCvsdata.GetYaxis().SetTitleFont(43)
    hratio_MCvsdata.GetYaxis().SetTitleOffset(1.0)
    hratio_MCvsdata.GetYaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    hratio_MCvsdata.GetYaxis().SetLabelSize(25)
    #X axis ratio plot settings
    hratio_MCvsdata.GetXaxis().SetTitleSize(25)
    hratio_MCvsdata.GetXaxis().SetTitleFont(43)
    hratio_MCvsdata.GetXaxis().SetTitleOffset(4.8)
    hratio_MCvsdata.GetXaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    hratio_MCvsdata.GetXaxis().SetLabelSize(20)
    hratio_MCvsdata.GetXaxis().SetLabelOffset(0.01)
    line_3In1 = r.TLine(hratio_MCvsdata.GetXaxis().GetXmin(),1,hratio_MCvsdata.GetXaxis().GetXmax(), 1)
    line_3In1.SetLineStyle(3)
    line_3In1.Draw()

    c.SaveAs('Complots/{}new/{}.png'.format(year,varname))

for v in columns_draw: plot_var(v)

#def plot_weight():
#    rMC_rw.Draw("MCw>>hMCw","{}".format(selMC+selBmass+selMCtruth))

#plot_weight()
    






