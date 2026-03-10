# A Temporal Binning Evaluation Method for Traffic Accident Anticipation


## Abstract
<p align="center">
  <img src="./comp_earliness_previous_ours.png" width="600" alt="Comparison of earliness evaluation methods">
</p>
<p>
    <em><b>Fig.1</b> Comparison of model earliness measurements: Previous methods measure earliness as the time between when the accident probability first exceeds the threshold and the accident. However, false positives can cause too early threshold crossings, where the measured earliness (60 frames) does not reflect the actual earliness (30 frames). In contrast, we measure earliness via prediction accuracy across temporal bins within a prediction window of length α, accounting for false positives.</em>
</p>
Effective collision prediction is critical for autonomous driving safety, yet existing Traffic Accident Anticipation (TAA) methods have fundamental limitations. They predict
per-frame accident probabilities and measure earliness using thresholds. Fig. 1 shows that this threshold-based evaluation does not account for false positives, which inflate earliness, providing false earliness estimates. Rather than presenting a new state-of-the-art architecture, we address these limitations by introducing a temporal binning evaluation method, which measures accuracy across temporal bins within a prediction window and accounts for false positives. This evaluation method requires models to predict when collisions will occur, a fundamentally different objective from traditional TAA. Therefore, to identify the best training strategy, we compare three task formulations – classification, regression, and a hybrid approach – using a VideoMAE [[1]](#1) backbone and test these on DADA-2000 [[2]](#2). Results show that the classification model achieves the highest mean bin accuracy across all prediction window lengths (1-5s), confirming classification as the best formulation. Starting 2.0s before collision, the classification model can predict collisions with a useful accuracy (better than random), though this only holds for VideoMAE on DADA-2000, with marginal improvement beyond 0.5s. Furthermore, separating normal frames from collision soon frames through a time-to-collision boundary (α) during training is essential. Without this boundary, the models fail to make
meaningful predictions

## Model variants
We study three task formulations on top of the same VideoMAE backbone:

- **Classification:** Predicts a discrete temporal bin label (No Collision Soon, Bin 5, …, Bin 1) for each frame.
- **Regression:** Predicts a continuous time-to-collision (TTC) in seconds for each frame.
- **Hybrid:** Uses a shared VideoMAE backbone with two heads; for each frame, the regression head predicts TTC in seconds and the binary head predicts Collision Soon vs No Collision Soon.

For regression and the hybrid approach, we explore variants with and without uncertainty prediction. For the precise labeling scheme and bin definitions, see Fig. 2 and Section 3 of the paper.

## Installation
Install dependencies using the provided `environment.yaml` file
```
conda env create -f environment.yaml
conda activate vmae
```

## Data preparation
Download the [DADA-2000](https://github.com/JWFangit/LOTVS-DADA) dataset and extract all zip files. Then, download the [annotations](https://huggingface.co/tue-mps/simple-tad/blob/main/datasets/D2K.zip) and extract them into the `DADA-2000` folder. Insert the `new_training.txt` and `new_validation.txt` files into the `DADA2K_my_split` to use our custom split (more details in the paper). The following file structure should be created. 

```
DADA2000
└───annotation 
└───DADA2K_my_split
│   │   new_training.txt
│   │   new_validation.txt
│   |   ...
│   
└───frames
    └───1 # accident category
        └───001 # video belonging to this category
            └─── images
                └───001.png # video frame
                └───...
            └─── ...
        └───...
        └───053
    │   ...
    └───61
```
## Usage

To train a models, please use the following command with {`classification`, `regression`, `hybrid`}.yaml:

```bash
python main.py fit \
    -c classification.yaml \
    --root /path/to/folder/with/datasets \
    --alpha 6 \
    --bin_width 1 
```

For the regression and hybrid models, you can optionally enable uncertainty prediction:
```bash
python main.py fit \
    -c classification.yaml \
    --root /path/to/folder/with/datasets \
    --alpha 6 \
    --bin_width 1 \
    --uncertainty_pred True
```

To validate a model, please use the following command:

```bash
python main.py validate \
-c classification.yaml \
--root /path/to/folder/with/datasets \
--alpha 6 \
--bin_width 1 \
--model_ckpt /path/to/saved_weights
```

**Important:**

- `--alpha` must be an integer (prediction horizon in seconds), e.g. `--alpha 6`.
- `--bin_width` is specified in **seconds** (can be fractional), e.g. `--bin_width 0.5`.
- The pair (`alpha`, `bin_width`) must provide an integer number of bins: `alpha / bin_width ∈ ℕ`.
- `bin_width` must correspond to an integer number of frames for the dataset's fps (30 fps for DADA-2000)

### Configuration

Additional training hyperparameters can be adjusted in the YAML configuration files (`classification.yaml`, `regression.yaml`, `hybrid.yaml`):

- **Learning rate**: Modify `optimizer.init_args.lr` (default: 1e-5 and 2e-5 for classsification and regression/hybrid, respectively)
- **Batch size**: Adjust `data.init_args.batch_size` (default: 32)
- **Number of frames per sliding window**: Modify `data.init_args.num_frames` (default: 16)
- **Sliding window FPS**: Adjust `data.init_args.sliding_window_fps` (default: 10)
- **Weighted sampling**: Enable bin-balanced sampling with `data.init_args.balancing: true` to oversample underrepresented bins. The WeightedRandomSampler in `dashcamaccidentdatamodule.py` 
  uses a fixed seed to ensure the reported results are reproducible.


## References
<a id="1">[1]</a> 
Z. Tong, Y. Song, J. Wang, and L. Wang, "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training," in *Advances in Neural Information Processing Systems* (NeurIPS), vol. 35, pp. 10078-10093, 2022.

<a id="2">[2]</a> 
J. Fang, D. Yan, J. Qiao, J. Xue, H. Wang and S. Li, "DADA-2000: Can Driving Accident be Predicted by Driver Attention? Analyzed by A Benchmark," in *IEEE Intelligent Transportation Systems Conference* (ITSC), pp. 4303-4309, 2019