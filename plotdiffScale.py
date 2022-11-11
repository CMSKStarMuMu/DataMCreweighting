from ROOT import TTree,TFile,TChain
import root_pandas
import numpy as np
import pandas as pd
import root_numpy
from array import array
pd.options.mode.chained_assignment = None  # default='warn'
import sys
import ROOT as r
#q2Bin = int(sys.argv[1])
parity = int(sys.argv[1])
year = int(sys.argv[2])
mc_sigma = 0.040
mc_mass  = 5.27783 
JPsiMass_ = 3.096916
nSigma_psiRej = 3.
selData = '( pass_preselection ==1 ) && \
            (abs(mumuMass - {JPSIM}) < {CUT}*mumuMassE) &&  eventN%2=={Parity}'\
            .format( JPSIM=JPsiMass_, CUT=nSigma_psiRej, Parity=parity)
selMC = selData + ' && (trig==1)' 
selMCtruth = '&& truthMatchMum == 1 && truthMatchMup == 1 && truthMatchTrkm == 1 && truthMatchTrkp == 1'
variables = {}
class variable:

    def __init__(self, name, title, binning=None):
        if binning is not None and (not isinstance(binning, list) or len(binning)!=4):
            raise Exception("Error in declaration of variable {0}. Binning must be a list like [nbins, min, max, SF]".format(name))
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

    def get_SF(self):
        if self.binning is not None:
            return self.binning[3]
        else: return 0



nBins = 200
ratiokk=[]
if (year==2016):
    ratiokk = [1.12567,1.11444]
elif (year==2017):
    ratiokk=[1.25446,1.22703]
else:
    ratiokk=[1.06216,1.08311]
variable("bLBSE"," bLBSE",[nBins, 0, 0.025,ratiokk[0]])
variable("bDCABSE","bDCABSE",[nBins, 0, 0.005,ratiokk[1]])
#variable("kstTrk1DCABSE","kstTrk1DCABSE",[nBins, 0., 3.2,1.])
#variable("kstTrk2DCABSE","kstTrk2DCABSE",[nBins, 0, 3.2,1.])
variable("kstTrk1DCABSE","kstTrk1DCABSE",[nBins, 0,0.025 ,1.])
variable("kstTrk2DCABSE","kstTrk2DCABSE",[nBins, 0,0.035,1.])
variable("kstTrkmDCABSE","kstTrkmDCABSE",[nBins, 0,0.035 ,1.])
variable("kstTrkpDCABSE","kstTrkpDCABSE",[nBins, 0,0.035,1.])
#varnames = ['bLBSE','bDCABSE','kstTrk1DCABSE','kstTrk2DCABSE']
varnames = ['bLBSE','bDCABSE']
#varnames = ['kstTrk1DCABSE','kstTrk2DCABSE']
#varnames = ["kstTrkpDCABSE","kstTrkmDCABSE"]

for varname in varnames:
    print ("Calculate and plot variable ", varname)

    maxi = variables[varname].get_xmax()
    mini = variables[varname].get_xmin()
    ratio = [1,variables[varname].get_SF()]
    print ("SF is", ratio)
    label = ['',' after scaled']
    color = [r.kRed,r.kBlue]

    rdata = TChain("ntuple")
    rMC = TChain("ntuple")
    rdata.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/preselection/data_charmonium_{}_preBDT_asSplot.root".format(year))
    rsw = TChain("fulldata")
    if (year ==2016):
        rsw.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/preselection/splot_weights_JPSI_2016B_2016H_L1all_preBDT.root")
    elif (year ==2017):
        rsw.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/preselection/splot_weights_JPSI_2017B_2017F_L1all_preBDT.root")
    else:
        rsw.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/preselection/splot_weights_JPSI_2018A_2018D_L1all_preBDT.root")
    rdata.AddFriend(rsw)
    rMC.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/preselection/MC_JPSI_{}_preBDT.root".format(year))

    if ('kstTrk1' in varname ):
        print ('data max is ', rdata.GetMaximum('kstTrkmDCABSE'),rdata.GetMaximum('kstTrkpDCABSE'))
        print ('MC max is ', rMC.GetMaximum('kstTrkmDCABSE'),rMC.GetMaximum('kstTrkpDCABSE'))
        
    else :
        print ('data max is ', rdata.GetMaximum(varname))
        print ('MC max is ', rMC.GetMaximum(varname))
    hdata = r.TH1F('h{}_data'.format(varname),'h{}_data'.format(varname),nBins,mini,maxi)
    
    if ('kstTrk' in varname ):
        select = '>' if varname=='kstTrk1DCABSE' else '<'
        #hdata = r.TH1F('h{}_data1'.format(varname),'h{}_data1'.format(varname),nBins,mini,maxi)
        rdata.Draw('kstTrkmDCABSE>>h{}_data'.format(varname),'nsig_sw*((kstTrkmPt{}kstTrkpPt)&&({}))'.format(select,selData),'goff')
        hdata2 = r.TH1F('h{}_data2'.format(varname),'h{}_data2'.format(varname),nBins,mini,maxi)
        rdata.Draw('kstTrkpDCABSE>>h{}_data2'.format(varname),'nsig_sw*((kstTrkpPt{}kstTrkmPt)&&({}))'.format(select,selData),'goff')
        hdata.Add(hdata2,1)
    else :
        rdata.Draw('{}>>h{}_data'.format(varname,varname),'nsig_sw*({})'.format(selData),'goff')
    prob=np.array([0.5])
    qdata=np.array([0.])
    y=hdata.GetQuantiles(1,qdata,prob)
    print ("data integral :" ,hdata.Integral())
    hdata.Scale(1./hdata.Integral())



    hMC = []
    hratio = []
    for i in range(0,2):
        
        #hMC.append(r.TH1F('h{}_MC_{}'.format(varname,label[i]),'h{}_MC_{}'.format(varname,label[i]),nBins,mini,maxi))
        if ('kstTrk' in varname ):  
            select = '>' if varname=='kstTrk1DCABSE' else '<'
            hMC1 = r.TH1F('h{}_MC_{}'.format(varname,label[i]),'h{}_MC_{}'.format(varname,label[i]),nBins,mini,maxi)
            rMC.Draw('kstTrkmDCABSE*{}>>h{}_MC_{}'.format(ratio[i],varname,label[i]),'weight*((kstTrkmPt{}kstTrkpPt)&&({}))'.format(select,selMC+selMCtruth),'goff')
            hMC2 = r.TH1F('h{}_MC_{}2'.format(varname,label[i]),'h{}_MC_{}2'.format(varname,label[i]),nBins,mini,maxi)
            rMC.Draw('kstTrkpDCABSE*{}>>h{}_MC_{}2'.format(ratio[i],varname,label[i]),'weight*((kstTrkpPt{}kstTrkmPt)&&({}))'.format(select,selMC+selMCtruth),'goff')
            hMC1.Add(hMC2,1)
            hMC.append(hMC1)
        else :
            hMC.append(r.TH1F('h{}_MC_{}'.format(varname,label[i]),'h{}_MC_{}'.format(varname,label[i]),nBins,mini,maxi))
            rMC.Draw('{}*{}>>h{}_MC_{}'.format(varname,ratio[i],varname,label[i]),'weight*({})'.format(selMC+selMCtruth),'goff')
        qMC=np.array([0.]) 
        y=hMC[i].GetQuantiles(1,qMC,prob)    
        print ("data median " , qdata[0])  
        print ("mc median " , qMC[0]) 
        print ("SF ", qdata[0]/qMC[0] )
        
        #print (hMC.KolmogorovTest(hdata[0],"DU"))
        print ("i is ", i , "MC integral :" ,hMC[i].Integral())
        hMC[i].Scale(1./hMC[i].Integral())

        #print (hMC.KolmogorovTest(hdata[0],""))
        print ("KStest of {} _ {} is \n {}".format(varname,label[i], hMC[i].KolmogorovTest(hdata,"D")))
        print ("Chi2 test of {} _ {} is \n {}".format(varname,label[i], hMC[i].Chi2Test(hdata,"WWP")))
        hratio.append(hMC[i].Clone('h{}_ratio_{}'.format(varname,label[i])))
        hratio[i].Divide(hdata)
        #hMC.Scale(1./hMC.Integral())

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
    #datapeak = hdata.GetBinCenter(hdata.GetMaximumBin())
    #print ('data peak is ', datapeak )
    hdata.SetMarkerColor(r.kBlack)
    hs.Add(hdata, "P")
    for i in range(0,2):
        hMC[i].SetLineColor(color[i])
        hMC[i].SetLineWidth(2)
        #MCpeak = hMC[i].GetBinCenter(hMC[i].GetMaximumBin())
        #print ('i =',i ,'  MCpeak is ', MCpeak )
        #SF = datapeak/MCpeak
        #print ('SF = ', SF)
        hs.Add(hMC[i],"hist")
        print('KS value of {} scaled of {} is {}'.format(label[i],varname,hdata.KolmogorovTest(hMC[i],"")))
    hs.Draw("nostack")
    hs.SetTitle("Plot of {}".format(varname))

    #Y axis h1 plot settings
    hs.GetYaxis().SetTitleSize(30)
    hs.GetYaxis().SetTitleFont(43)
    hs.GetYaxis().SetTitleOffset(0.5)
    hs.GetYaxis().SetLabelFont(43) #Absolute font size in pixel (precision 3)
    hs.GetYaxis().SetLabelSize(25)
    hs.GetXaxis().SetTitleSize(0)
    hs.GetXaxis().SetLabelSize(20)

    leg = r.TLegend(0.7,0.7,0.9,0.9)
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
        hratio[i].SetLineWidth(2)
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

    c.SaveAs('plots/medianSF_new/Scale_{}_{}.png'.format(varname,year))






