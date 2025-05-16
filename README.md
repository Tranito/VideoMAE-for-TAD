# vitin1k
1. ```conda create -n vitin1k python=3.10 && conda activate vitin1k```
2. ```pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1```
3. ```pip install -r requirements.txt```
4. ```conda install nvidia/label/cuda-12.4.0::cuda```
5. ```python main.py {fit,validate,test,predict} -c config.yaml --root /path/to/folder/with/datasets```
