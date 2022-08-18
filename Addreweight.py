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

rMC = TChain("ntuple")
rMC.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}_MC_JPSI_scale_and_preselection_p{}.root".format(year,parity))



ds = pd.DataFrame(
    root_numpy.tree2array(
        tree = rMC
    )
)
dw = root_pandas.read_root("./JPsiK_reweight_{}.root".format(year))
dw['MCw']=dw['MCw'].astype(float)
data_final = pd.merge(ds,dw,left_index=True,right_index=True)
ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/{}_MC_JPSI_scale_and_preselection_XGBV4_p{}.root".format(year,parity)
data_final.to_root(ofile, key='ntuple', store_index=False)