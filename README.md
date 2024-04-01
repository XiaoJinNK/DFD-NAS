## Introduction

**This code is based on the implementation of  [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).**

## Results
### Comparisons of the in-dataset results on Celeb-DF, Wild-Deepfake, and DFDC-preview datasets.
Method | Celeb-DF <br/>AUC | WildDeepfake<br/>AUC | DFDC-pre<br/>AUC
--- |---------------|----------------------| ---
Meso4| 66.17| 66.5 |76.47
RNN  | 86.52| 67.35 | 77.48
FWA |  60.16|  57.92|72.97
Xception  |  89.75| 80.89|81.58
FT-TS |  86.67|  68.09 | 64.03
FInfer|93.30|81.38|82.88
Ours_in |  99.59|   85.45 |  88.72
Ours_cross |   99.75|  86.41 | 88.70

###Comparisons of the cross-dataset results on WildDeep-fake, and DFDC-preview datasets. These methods are trained on FF++.
Method | WildDeepfake<br/>AUC | DFDC-pre<br/>AUC
--- | --- | --- 
Meso4| 59.74|59.30
RNN  |67.03|59.37
FWA | 67.35|59.49
Xception  |60.54|64.29
FT-TS | 59.82|59.09
FInfer|64.31|69.06
RECCE |   99.75|  86.41 
Ours_in | 74.32|74.85 
Ours_cross |74.71|75.79
## Usage
#### Search Stage 

To run our code, you only need one Nvidia 3080ti(12G memory).


In dataset search

```
python train_search_theta.py \\
```

Cross dataset search
```
python train_search_cross_domain.py \\
```

#### Evaluation Stage 

```
python train.py \\
```
#### Test Stage 

```
python test.py \\
```
## Pretrained models
You can load the pre-trained weights in the 'indataset_pth_txt/eval-EXP-cross_domain_train_Celeb-DF2/pth' folder.

## Notes
- For the codes in the main branch, `python3 with pytorch(1.12.1)` is recommended ï¼ˆrunning on `Nvidia 3080ti`ï¼? 

- We provide pre-trained weights on Celeb-DF2.
You can load the pre-trained weights in the 'indataset_pth_txt/eval-EXP-cross_domain_train_Celeb-DF2/pth' folder.
Run test.py to complete the test, and the results correspond to 'ours_cross' in table1 of the paper. 
Due to the different pre-processing methods of videos and the randomness of the selected video frames in the test dataset, there may be slight differences in the test results.
The experimental records are stored in the 'test_result/' folder.
## Related work
[PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS)

[EoiNAS](https://github.com/ZLab540/Exploiting-Operation-Importance-for-Differentiable-Neural-Architecture-Search/tree/main/EoiNAS_image%20classification)
