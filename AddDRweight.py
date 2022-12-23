from ROOT import TTree,TFile,TChain
import root_pandas
import numpy as np
import pandas as pd
import root_numpy
from array import array
pd.options.mode.chained_assignment = None  # default='warn'
import sys

year = int(sys.argv[1])

rMC = TChain("ntuple")
rMC.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/after_nominal_selection/{}MC_JPSI_noIP2D_addxcutvariable.root".format(year))
print ("rMC entries: ", rMC.GetEntries())


ds = pd.DataFrame(
    root_numpy.tree2array(
        tree = rMC
    )
)
dw = root_pandas.read_root("/afs/cern.ch/work/d/dini/public/{}MC_JPSI_PDGinputs.root".format(year))
print ("dw size: ", dw.shape[0] )
dw['DRweight']=dw['DRweight'].astype(float)
data_final = pd.merge(ds,dw,left_index=True,right_index=True)


ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGB_postBDT/{}MC_JPSI_forXGB_AddDRweight.root".format(year)
data_final.to_root(ofile, key='ntuple', store_index=False)