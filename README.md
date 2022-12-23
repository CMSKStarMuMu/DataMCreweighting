# DataMCreweighting
Reweighting code of data/MC difference

## Merge DRweight with ntuple

```sh
python AddDRweight.py 2016
```

## Generate model

We just use even parity Jpsi channel to generate reweighting model

```sh
python DoReweight_2class.py 4 0 2016 10 10 7 100
python DoReweight_3class.py 4 0 2016 10 10 7 1000
```
With early stop, best iteration number for 2 class is 40, for 3 class is 102 (will be different due to different train/test sample split)

## Apply model

Then we can apply model

```sh
python ApplyReweight_2class.py 4 1 2016
python ApplyReweight_3class.py 4 1 2016
```

And we can also get MCprob for data and MC samples
```sh
python ApplyReweight_getBDT_2class.py 4 1 2016 0
python ApplyReweight_getBDT_2class.py 4 1 2016 1
```

## Draw comparison plots
```sh
plotdiffoptionbin4_2class.py 4 1 2016
plotdiffoptionbin4_3class.py 4 1 2016
```




