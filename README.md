# DataMCreweighting 

Remember to check the parity if you want to change the parity, sometimes the parity configuration in the command line is useless in this version of code

## Addnewsw.py
Use this code to merge positive sweights with original data ntuple


## DoReweight_noBmass.py

```sh
python DoReweight_noBmass.py q2Bin parity year eta subsample max_depth num_round
```
True eta in the code is eta/100, true subsample is subsample/10

Recommendation for the parameters: 
```sh
python DoReweight_noBmass.py 4 0 2016 3 3 4 150
```

## ApplyReweight.py
Recommendation for the parameters: 
```sh
python ApplyReweighy.py 4 1 2016
```


## ApplyReweight_getBDT.py

```sh
python ApplyReweight_getBDT.py 4 1 2016 0
```
Change final parameter: 0 means MC, 1 means data


## plotdiffoptionbin4_BDT_forBpag_check.py
```sh
python plotdiffoptionbin4_BDT_forBpag_check.py 4 1 2016
```
plot BDT output comparison, you can just AddFriendTree of MCweights and BDToutput from ApplyReweight code to plot the comparison. And also, as MCweights= BDTout/(1-BDTout), we only need one for MC samples
