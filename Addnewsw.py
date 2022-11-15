from ROOT import TTree,TFile,TChain
import root_pandas
import numpy as np
import pandas as pd
import root_numpy
from array import array
pd.options.mode.chained_assignment = None  # default='warn'
import sys

q2Bin = int(sys.argv[1])
year = int(sys.argv[2])

rdata = TChain("ntuple")
rdata.Add("~/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_data_beforsel.root".format(year,year))
print ("rMC entries: ", rdata.GetEntries())


ds = pd.DataFrame(
    root_numpy.tree2array(
        tree = rdata
    )
)
dw = root_pandas.read_root("/eos/user/a/aboletti/BdToKstarMuMu/sWeightsConversion/{}_data_beforsel_posWei_div6.root".format(year))
print ("dw size: ", dw.shape[0] )
dw['nsig_sw_pos']=dw['nsig_sw_pos'].astype(float)
data_final = pd.merge(ds,dw,left_index=True,right_index=True)

'''db = root_pandas.read_root("./JPsiKMC_BDTout_XGBV5_2016_p1_new.root")
print ("db size: ", db.shape[0] )
db['BDTout']=db['BDTout'].astype(float)
data_final1 = pd.merge(data_final,db,left_index=True,right_index=True)'''


ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV5/{}/{}_data_beforsel_posWei_div6.root".format(year,year)
data_final.to_root(ofile, key='ntuple', store_index=False)