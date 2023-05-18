# Flow-for-Trajectory-prediction

Apply **masked autoregressive flow** on human trajectory prediction task.

The structure of MAF follow the settings of flow-net part of LDS(https://arxiv.org/abs/2003.03212)

## Data
Use the data preprocessed by Ynet(http://arxiv.org/abs/2012.01526)

## Usage
- simple run
```bash
python main.py -e 100 -b 32 --lr 0.002 -d 0 --des "b32_lr0.002"
```
