# DataMCreweighting 

If you want to do simple check, please use the code with "test" in the name, which same as without "test" code but different input variables for reweighting

## DoReweight_noBmass.py

```sh
python DoReweight_noBmass.py q2Bin parity year eta subsample max_depth num_round
```
True eta in the code is eta/100, true subsample is subsample/10

Recommendation for the parameters: 
```sh
python DoReweight_noBmass.py 4 0 2016 5 5 5 200
```
If simple check
```sh
python DoReweight_test.py 4 0 2016 3 5 5 200
```

For DoReweight_noBmass.py
L180-L182: change the weights for data and MC
L301 and others: change the folder of XGB performance plots
L519 : change the model output

## ApplyReweight.py
Recommendation for the parameters: 
```sh
python DoReweight_noBmass.py 4 1 2016
```
L242: input model json file
L262: output MC weights
remember to keep reweighting variables and PUweights consistent with DoReweight step

## ApplyReweight_getBDT.py

```sh
python ApplyReweight_getBDT.py 4 1 2016
```
similar as the code to get MC weights. But we can get BDToutput for MC and data. Remember to keep sweights and PUweights consistent with input.

L251: xg_data_only: give data BDToutput, xg_phsp_only: give MC BDToutput

## plotdiffoptionbin4.py

```sh
python plotdiffoptionbin4.py 4 1 2016
```
Plot the comparison plots

## plotdiffoptionbin4_BDT_forBpag.py
```sh
python plotdiffoptionbin4.py 4 1 2016
```
plot BDT output comparison, you can just AddFriendTree of MCweights and BDToutput from ApplyReweight code to plot the comparison. And also, as MCweights= BDTout/(1-BDTout), we only need one for MC samples
