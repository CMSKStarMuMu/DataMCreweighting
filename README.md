# DataMCreweighting
Reweighting XGBV4 code of data/MC difference

## Create Dataset

Create MC dataset, add needed branch and selection (selection only to choose events for generate reweighting model)

```sh
python SelectMC.py parity
```

## Generate model

We just use even parity Jpsi channel to generate reweighting model

```sh
python DoReweight.py q2Bin parity year
```

## Apply model

Then we can apply model to each q2Bin (here use odd parity events) 

```sh
python ApplyReweight.py q2Bin parity year
```

## Compare variables distributions
remember that we can only compare data and MC distributions in Jpsi channel and Psi(2S) channel

```sh      
python plotdiffoptionbin4.py
```