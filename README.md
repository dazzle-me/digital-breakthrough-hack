# digital-breakthrough-hack
N-th place solution of RZD hackathon

Note: to reproduce solution you should have a machine with at least 2x3090

## Download data

[train](https://lodmedia.hb.bizmrg.com/case_files/766370/train_dataset_train.zip), [test](https://lodmedia.hb.bizmrg.com/case_files/766370/test_dataset_test.zip)

## Data dir tree structure
```
-- data
    |-- test [1000 entries exceeds filelimit, not opening dir]
    |-- train
    |   |-- images [8203 entries exceeds filelimit, not opening dir]
    |   `- mask [8203 entries exceeds filelimit, not opening dir]
```

## Prepare env
``` 
cd mmsegmentation/docker
docker build . -t rlh
```

## Launch docker
```
bash launch_docker.sh
```

## Prepare training masks & split
```
cd /workspace/rlh
python3 create_split.py
python3 create_masks.py
```

## Data dir structure now

```
-- data
    |-- stratified_split
    |   |-- test.txt
    |   |-- train.txt
    |   `-- val.txt
    |-- test [1000 entries exceeds filelimit, not opening dir]
    |-- train
    |   |-- images [8203 entries exceeds filelimit, not opening dir]
    |   |-- mask [8203 entries exceeds filelimit, not opening dir]
    |   |-- meta.csv
    |   `-- new_mask [8203 entries exceeds filelimit, not opening dir]
```


## Train model

```
cd /workspace/rlh/mmsegmentation
bash tools/dist_train.sh /workspace/rlh/configs/config_27_augs.py 2 --work-dir /workspace/rlh/work_dirs/exp37
```

## Inference trained model
```
python3 inference.py --exp exp37
```
