from ROOT import TChain, TFile, TTree
import root_pandas
import pandas as pd
from math import sin, cos, sqrt, pi
import sys
import numpy as np

q2Bin = int(sys.argv[1])
parity = int(sys.argv[2])
year = int(sys.argv[3])
selection = bool(int(sys.argv[4]))
usePDG = bool(int(sys.argv[5]))

print ('q2Bin, ',q2Bin)
print ('parity ', parity)
print ('year ', year)
print ('selection ', selection)
print ('usePDG ', usePDG)


def getParWeight(ctK,ctL,phi,usePDG):
    Fl  = 0.5534
    P1  = -0.0159
    P2  = -0.0012
    P3  = 0.237
    P4p = -0.9557
    P5p = -0.0064
    P6p = 0.0018
    P8p = -0.218

    FlMC =  5.9999e-01
    P1MC = -1.9816e-01
    P2MC = -2.7908e-04
    P3MC = -4.4279e-04
    P4pMC = -8.7036e-01
    P5pMC =  1.1506e-03
    P6pMC = -3.2764e-04
    P8pMC =  4.2577e-04

    FlPDG = 0.571

    FlVal = Fl
    P2Val = P2
    if (usePDG):
        FlVal = FlPDG
        P2Val = 0
        

    decMC = ( 0.75 * (1-FlMC) * (1-ctK*ctK) +
            FlMC * ctK*ctK +
            ( 0.25 * (1-FlMC) * (1-ctK*ctK) - FlMC * ctK*ctK ) * ( 2 * ctL*ctL -1 ) +
            0.5 * P1MC * (1-FlMC) * (1-ctK*ctK) * (1-ctL*ctL) * cos(2*phi) +
            2 * cos(phi) * ctK * sqrt(FlMC * (1-FlMC) * (1-ctK*ctK)) * ( P4pMC * ctL * sqrt(1-ctL*ctL) + P5pMC * sqrt(1-ctL*ctL) ) +
            2 * sin(phi) * ctK * sqrt(FlMC * (1-FlMC) * (1-ctK*ctK)) * ( P8pMC * ctL * sqrt(1-ctL*ctL) - P6pMC * sqrt(1-ctL*ctL) ) +
            2 * P2MC * (1-FlMC) * (1-ctK*ctK) * ctL -
            P3MC * (1-FlMC) * (1-ctK*ctK) * (1-ctL*ctL) * sin(2*phi) )
    
    #print ('decMC ', decMC)

    decTarget = ( 0.75 * (1-FlVal) * (1-ctK*ctK) +
                FlVal * ctK*ctK +
                ( 0.25 * (1-FlVal) * (1-ctK*ctK) - FlVal * ctK*ctK ) * ( 2 * ctL*ctL -1 ) +
                0.5 * P1 * (1-FlVal) * (1-ctK*ctK) * (1-ctL*ctL) * cos(2*phi) +
                2 * cos(phi) * ctK * sqrt(FlVal * (1-FlVal) * (1-ctK*ctK)) * ( P4p * ctL * sqrt(1-ctL*ctL) + P5p * sqrt(1-ctL*ctL) ) +
                2 * sin(phi) * ctK * sqrt(FlVal * (1-FlVal) * (1-ctK*ctK)) * ( P8p * ctL * sqrt(1-ctL*ctL) - P6p * sqrt(1-ctL*ctL) ) +
                2 * P2Val * (1-FlVal) * (1-ctK*ctK) * ctL -
                P3 * (1-FlVal) * (1-ctK*ctK) * (1-ctL*ctL) * sin(2*phi) )
    #print ('decTarget ', decTarget)
    

    return decTarget/decMC


t = TChain("ntuple")
if (selection):
    t.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}/{}_MC_JPSI_scale_and_preselection_forreweighting_p{}.root".format(year,year,parity))
else:
    t.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}/{}_MC_JPSI_scale_and_preselection_p{}.root".format(year,year,parity))
    
nEntry = t.GetEntries()
ds_DCrate = pd.DataFrame(index=range(nEntry),columns=['DCratew'],dtype=np.float64)
for i in range(0, nEntry):
    t.GetEntry(i)
    ctK = t.gen_cos_theta_l
    ctL = t.gen_cos_theta_k
    phi = t.gen_phi_kst_mumu
    if (abs(ctK)>1 or abs(ctL>1) or abs(phi)>pi):
        print ('i = ', i,' 3 angles ', ctK, '', ctL,'', phi)
        weight=1
        ds_DCrate.loc[i,'DCratew']=weight
        
    else:
        weight = getParWeight(ctK=ctK,ctL=ctL,phi=phi,usePDG=usePDG)
        ds_DCrate.loc[i,'DCratew']=weight
    
ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}/{}_MC_JPSI_DCratew_p{}_sel{}_PDGv{}.root".format(year,year,parity,selection,usePDG)

ds_DCrate.to_root(ofile, key='ntuple', store_index=False)


    
    