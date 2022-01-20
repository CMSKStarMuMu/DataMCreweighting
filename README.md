# DataMCreweighting
Reweighting code of data/MC difference

## Create Dataset

```sh
root -l 'createDataset.cc(year,q2Bin,data,deno,num)'
```
Then we can generate part of data/MC ntuple after all selections (due to limitations of ntuple size)

Then add sWeights to data ntuple, merge all files and divide into two parities

```sh
python AddSweight.py
```
For MC we just need to merge all root files

```sh
python MCmerge.py
```

## Generate model

We just use even parity Jpsi channel to generate reweighting model

```sh
python DoReweight_readindat_fulldatatraining.py q2Bin parity year
```

## Apply model

Then we can apply model to each q2Bin (here use odd parity events) 

```sh
python Application_DoReweight_readindat_fulldatatraining.py
```
remember that we can only compare data and MC distributions in Jpsi channel and Psi(2S) channel

