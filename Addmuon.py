from ROOT import TTree,TFile,TChain
import root_pandas
import numpy as np
import pandas as pd
import root_numpy
from array import array
pd.options.mode.chained_assignment = None  # default='warn'
import sys

q2Bin = int(sys.argv[1])
parity = int(sys.argv[2])
year = int(sys.argv[3])
selection = bool(int(sys.argv[4]))


t = TChain("ntuple")
if (selection):
    t.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}/{}_MC_JPSI_scale_and_preselection_forreweighting_p{}.root".format(year,year,parity))
else:
    t.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}/{}_MC_JPSI_scale_and_preselection_p{}.root".format(year,year,parity))


r = TChain("ntuple")
usePDG = True

r.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}/{}_MC_JPSI_DCratew_p{}_sel{}_PDGv{}.root".format(year,year,parity,selection,usePDG))
    
ds = pd.DataFrame(
    root_numpy.tree2array(
        tree = t
    )
)

entries = len(ds)

ds['mu1Pt']=0
ds['mu1Eta']=0
ds['mu1Phi']=0

ds['mu2Pt']=0
ds['mu2Eta']=0
ds['mu2Phi']=0

ds2 = pd.DataFrame(
    root_numpy.tree2array(
        tree = r
    )
)

entries2 = len(ds2)

if (entries!=entries2):
    print ("wrong number of DCRate Tree")
    exit(1)


for i in range(0,entries):
    if (i%100000==0):
        print ("process ", i)
    if (ds.loc[i,'mumPt']>ds.loc[i,'mupPt']):
        ds.loc[i,'mu1Pt']=ds.loc[i,'mumPt']
        ds.loc[i,'mu1Eta']=ds.loc[i,'mumEta']
        ds.loc[i,'mu1Phi']=ds.loc[i,'mumPhi']

        ds.loc[i,'mu2Pt']=ds.loc[i,'mupPt']
        ds.loc[i,'mu2Eta']=ds.loc[i,'mupEta']
        ds.loc[i,'mu2Phi']=ds.loc[i,'mupPhi']

    else:
        ds.loc[i,'mu2Pt']=ds.loc[i,'mumPt']
        ds.loc[i,'mu2Eta']=ds.loc[i,'mumEta']
        ds.loc[i,'mu2Phi']=ds.loc[i,'mumPhi']

        ds.loc[i,'mu1Pt']=ds.loc[i,'mupPt']
        ds.loc[i,'mu1Eta']=ds.loc[i,'mupEta']
        ds.loc[i,'mu1Phi']=ds.loc[i,'mupPhi']
        
        
ds = pd.concat([ds,ds2],axis=1)
        
ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_MC_JPSI_scale_and_preselection_p{}.root".format(year,year,parity)
print ('\t...done. n events: ', len(ds))
ds.to_root(ofile, key='ntuple', store_index=False)

    