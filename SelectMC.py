from ROOT import TTree,TFile,TChain
import root_pandas
import numpy as np
import pandas as pd
import root_numpy
from array import array
pd.options.mode.chained_assignment = None  # default='warn'
import sys

parity = int(sys.argv[1])
mc_sigma = 0.040
mc_mass  = 5.27783 
JPsiMass_ = 3.096916
nSigma_psiRej = 3.
'''selData = '( ( (bMass*tagB0 + (1-tagB0)*bBarMass)   > {M}-3*{S}   && \
            (bMass*tagB0 + (1-tagB0)*bBarMass)   < {M}+3*{S} ) && \
            ( pass_preselection ==1 ) && \
            (abs(mumuMass - {JPSIM}) < {CUT}*mumuMassE))'\
            .format(M=mc_mass,S=mc_sigma,  JPSIM=JPsiMass_, CUT=nSigma_psiRej)
selMC = selData + ' && (trig==1)'+ ' && (truthMatchMum == 1 && truthMatchMup == 1 && truthMatchTrkm == 1 && truthMatchTrkp == 1)'
'''
selData = ''
selMC = ''
r = TChain("ntuple")

r.Add("/eos/cms/store/group/phys_bphys/fiorendi/p5prime/ntuples/for_reweighting_V4/2018_MC_JPSI_scale_and_preselection.root")

print ("Events before sel:" , r.GetEntries())
ds = pd.DataFrame(
    root_numpy.tree2array(
        tree = r,
        selection="eventN%2=={}".format(parity)
        #stop=10000
    )
)


entries = len(ds)
print ("entries is ", entries)
ds['tagged_mass']=0
ds['kstTrk1Pt']=0
ds['kstTrk1MinIP2D']=0
ds['kstTrk1Eta']=0
ds['kstTrk1Phi']=0

ds['kstTrk2Pt']=0
ds['kstTrk2MinIP2D']=0
ds['kstTrk2Eta']=0
ds['kstTrk2Phi']=0



for i in range(0,entries):
    if (i%100000==0):
        print ("process ", i)
    if (ds.loc[i,'tagB0']==1):
        ds.loc[i,'tagged_mass']=ds.loc[i,'bMass']
    else:
        ds.loc[i,'tagged_mass']=ds.loc[i,'bBarMass']  

    if (ds.loc[i,'kstTrkmPt']>ds.loc[i,'kstTrkpPt']):
        ds.loc[i,'kstTrk1Pt']=ds.loc[i,'kstTrkmPt']
        ds.loc[i,'kstTrk1MinIP2D']=ds.loc[i,'kstTrkmMinIP2D']
        ds.loc[i,'kstTrk1Eta']=ds.loc[i,'kstTrkmEta']
        ds.loc[i,'kstTrk1Phi']=ds.loc[i,'kstTrkmPhi']

        ds.loc[i,'kstTrk2Pt']=ds.loc[i,'kstTrkpPt']
        ds.loc[i,'kstTrk2MinIP2D']=ds.loc[i,'kstTrkpMinIP2D']
        ds.loc[i,'kstTrk2Eta']=ds.loc[i,'kstTrkpEta']
        ds.loc[i,'kstTrk2Phi']=ds.loc[i,'kstTrkpPhi']

    else:
        ds.loc[i,'kstTrk2Pt']=ds.loc[i,'kstTrkmPt']
        ds.loc[i,'kstTrk2MinIP2D']=ds.loc[i,'kstTrkmMinIP2D']
        ds.loc[i,'kstTrk2Eta']=ds.loc[i,'kstTrkmEta']
        ds.loc[i,'kstTrk2Phi']=ds.loc[i,'kstTrkmPhi']

        ds.loc[i,'kstTrk1Pt']=ds.loc[i,'kstTrkpPt']
        ds.loc[i,'kstTrk1MinIP2D']=ds.loc[i,'kstTrkpMinIP2D']
        ds.loc[i,'kstTrk1Eta']=ds.loc[i,'kstTrkpEta']
        ds.loc[i,'kstTrk1Phi']=ds.loc[i,'kstTrkpPhi']
    


ofile = "/afs/cern.ch/user/x/xuqin/cernbox/workdir/B0KstMuMu/reweight/Tree/final/XGBV4/2018_MC_JPSI_scale_and_preselection_p{}.root".format(parity)
print ('\t...done. n events: ', len(ds))
ds.to_root(ofile, key='ntuple', store_index=False)


