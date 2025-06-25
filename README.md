

# Running an experiment
### 1. Dataset
Due to space limitations, we compressed the dataset. You can unzip all xxx.zip data in its fold.

## 2. Training ZSCAN-DDIE
The data folder contains CSV files that have been divided. I you need to download the required dependency packages from requirements.txt and execute the following command

```python
python main.py --config configs.py
```
## 3. Testing ZSCAN-DDIE
After training, the parameters of models are saved in ./work_dirs/

Then, you can test the model by:

```python
python main.py --config configs.py --zsl_para work_dirs/zscan_ddie/model_parameter/zsl_best_epoch50.pkl --gzsl_para work_dirs/zscan_ddie/model_parameter/gzsl_best_epoch50.pkl
```