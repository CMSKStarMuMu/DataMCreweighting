from ROOT import TTree,TFile,TChain
import root_pandas
import numpy as np
import pandas as pd
import root_numpy
from array import array
pd.options.mode.chained_assignment = None  # default='warn'

parity=1 
year=2018

rdata = TChain("ntuple")
if (year==2016):
    rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b6_2016_1_1.root')
elif (year==2017):
    rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b6_2017_1_1.root')
    #rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b4_2017_2_2.root')
else:
    rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b6_2018_1_1.root')
    #rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b4_2018_2_3.root')
    #rdata.Add('/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b4_2018_3_3.root')

sw = TChain("tree_sw")
sw.Add("/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/SWtree_{}_b6.root".format(year))

ds = pd.DataFrame(
    root_numpy.tree2array(
        tree = rdata
    )
)
dw = pd.DataFrame(
    root_numpy.tree2array(
        tree = sw
    )
)

data_final = pd.merge(ds,dw,left_index=True,right_index=True)
data_sel = data_final.query("eventN%2=={}".format(parity))
ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/finaltree/recoDATADataset_b6_{}_p{}.root".format(year,parity)
print ('\t...done. n events: ', len(data_sel))
data_sel.to_root(ofile, key='ntuple', store_index=False)

