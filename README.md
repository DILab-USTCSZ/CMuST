# Get Rid of Isolation: A Continuous Multi-task Spatio-Temporal Learning Framework

This folder concludes the further revised version of pytorch implementation of our CMuST model.

## Requirements

- python 3.8
- see `requirements.txt`

## Dataset Sources

### NYC dataset

Taxi Trip Records https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Demographic Data: https://data.cityofnewyork.us/City-Government/Demograp-hic-Statistics-By-Zip-Code/

Road Network: https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/

### SIP dataset

Due to the privacy protocols between our department and SIP traffic administration offices, the statistics of traffic flows and speed values cannot be open source.

### Chicago dataset

Taxi Trip Records https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew/about_data

Traffic Crashes - People https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d/about_data

Traffic Crashes - Crashes https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data

Traffic Crashes - Vehicles https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3/about_data

## Data Preparation

This can be found in `./data/README.md`.

## Training Example

```python
python main.py --dataset NYC --num_nodes 206 --tod_size 48 --gpu 0
```

```python
python main.py --dataset CHI --num_nodes 220 --tod_size 48 --gpu 1
```

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{yi2024get,
  title={Get Rid of Isolation: A Continuous Multi-task Spatio-Temporal Learning Framework},
  author={Yi, Zhongchao and Zhou, Zhengyang and Huang, Qihe and Chen, Yanjiang and Yu, Liheng and Wang, Xu and Wang, Yang},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024}
}
```

## Contact
If you have any questions or suggestions, please feel free to contact:
- Zhongchao Yi ([zhongchaoyi@mail.ustc.edu.cn]())
- Zhengyang Zhou ([zzy0929@ustc.edu.cn]())


## More Related Works

- [ComS2T: A complementary spatiotemporal learning system for data-adaptive model evolution. arXiv Preprint.](https://arxiv.org/pdf/2403.01738)

- [Maintaining the status quo: Capturing invariant relations for OOD spatiotemporal learning. KDD'23.](http://home.ustc.edu.cn/~zzy0929/Home/Paper/KDD23_CauSTG.pdf) [[Code]](https://github.com/zzyy0929/KDD23-CauSTG)

- [LeRet: Language-Empowered Retentive Network for Time Series Forecasting. IJCAI'24.](http://home.ustc.edu.cn/~zzy0929/Home/Paper/IJCAI24_LeRet.pdf) [[Code]](https://github.com/hqh0728/LeRet)

## Acknowledgement

We sincerely thanks the following GitHub repositories for providing valuable codebases and datasets:

https://github.com/liuxu77/LargeST

https://github.com/ACAT-SCUT/CycleNet

https://github.com/nnzhan/Graph-WaveNet

https://github.com/GestaltCogTeam/STID

https://github.com/XDZhelheim/STAEformer

https://github.com/Echohhhhhh/GSNet