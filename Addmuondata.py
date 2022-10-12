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


t = TChain("ntuple")

t.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}/{}_data_aftersel_p{}.root".format(year,year,parity))

    
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
        
        
ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_data_aftersel_p{}.root".format(year,year,parity)
print ('\t...done. n events: ', len(ds))
ds.to_root(ofile, key='ntuple', store_index=False)

    