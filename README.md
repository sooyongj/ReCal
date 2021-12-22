# Recursive  lossy label-invariant Calibration (ReCal)

This repository is for Improving Classifier Confidence using Lossy Label-Invariant Transformations [[1]](#1).
ReCal supports MNIST, CIFAR10/100, ImageNet with various models for each dataset, but currently, this repository does not contain the training code of models for MNIST and CIFAR10/100.
Therefore, you can use ReCal for ImageNet. 

# How to use
We provide two versions of ReCal: 1) library version, and 2) standalone version.
With the library version, you can apply ReCal to your dataset and model, and standalone version is for the selected datasets and models.

To apply ReCal, you need to do 1) Export logits, and 2) Calibrate and evaluate.
The below two sections will explain each steps and provide sample command.

## Library Version
With this library version, ReCal can be applied to any datasets and models.

### Export logits.
'ReCal/experiments/main_export_example.py' is an example to show how to use 'ReCal/LogitsFileExport.py.'
With this example, the logits for MNIST validation set will be stored as files.
The below command is how to use the script.`
```
python ReCal/experiments/main_export_logits.py
```


### Calibration
'ReCal/experiments/main_example.py' illustrates how to use 'ReCal/Calibrator.py.' This script loads the logits for MNIST validation set which prepared by the above 'main_export_example.py', and calibrate MNIST LeNet5 classifier for MNIST test set data.
The below command is how to use the script.`
```
python ReCal/experiments/main_example.py
```

## Standalone version
The following sections explains how to export logits and calibrate classifiers for the datasets and models like the paper. 
### Export logits
'main_export_logits.py' will export logits for the given dataset and model. ReCal uses multiple transformations for calibration, and this script will export all logits before and after such transformations.
The transformation types and arguments can be configured with the command line options, and please refer to the below table for the detail.
This script will compute logits before/after transformation and store the logits in 'outputs/DATASET/MODEL_ITERLIST_NTrans'. 

The below command is for extracting logits for DenseNet161 on ImageNet with 10 Zoom-out transformation whose scale is between 0.5, 0.9,
``` 
python ReCal/experiments/main_export_logits.py --dataset ImageNet --model densenet161
```

|Option|Desc|Default|Possible values|
|------|----|-------|---------------|
|--dataset|Specify the dataset|CIFAR10|MNIST, CIFAR10, CIFAR100, ImageNet|
|--model|Specify the model|densenet40|Depends on dataset, the full list is in the paper|
|--before_ts|Starts with uncalibrated confidence|||
|--after_ts|Starts with calibrated confidence (by Tempersature Scaling)||
|--trans_type|Specify the transformation type|zoom|zoom, brightness|
|--trans_arg|Specify the transformation factor|0.5|Float numbers|
|--train_batch_size|Specify the batch size for train set|100|Integer numbers|
|--test_batch_size|Specify the batch size for test set|100|Integer numbers|
|--seed|Specify random seed|100|Integer numbers|
|--device|Specify whether GPU is used or not|cuda|cuda, cpu|
|--iter_list|Specify the transformation specification|['zoom', 0.5, 0.9]|[Tran_type, Tran_arg_min, Tran_arg_max]|
|--n_trans|Number of transformations|10|Integer numbers|
 

### Calibrate and evaluate
After logits are prepared, 'main_logits.py' will load the logits and calibrate the model and evalute the result.
The below command is for ImageNet, DenseNet161 with 10 zoom-out transformations whose scale factor is between 0.5 and 0.9.

```
python ReCal/experiments/main_logits.py --dataset ImageNet --model densenet161 --root_dir outputs/ImageNet/densenet_zoom_0.5_0.9_10/uncalibrated --n_iters 200
```

|Option|Desc|Default|Possible values|
|------|----|-------|---------------|
|--dataset|Specify the dataset|MNIST|MNIST, CIFAR10, CIFAR100, ImageNet|
|--model|Specify the model|lenet5|Depends on dataset, the full list is in the paper|
|--cal_method|Specify whether the initial confidence is uncalibrated or not|uncalibrated|uncalibrated, ts|
|--root_dir|Specify the location of extracted logits|outputs/MNIST/lenet5_zoom_0.5_0.9_10/uncalibrated|
|--seed|Specify random seed|100|Integer numbers|
|--device|Specify whether GPU is used or not|cuda|cuda, cpu|
|--iter_list|Specify the transformation specification|['zoom', 0.5, 0.9]|[Tran_type, Tran_arg_min, Tran_arg_max]|
|--n_iters|Number of iterations|5|Integer numbers|


# Reference
<a id="1">[1]</a>
Jang S, Lee I, Weimer J. Improving Classifier Confidence using Lossy Label-Invariant Transformations. InInternational Conference on Artificial Intelligence and Statistics 2021 Mar 18 (pp. 4051-4059). PMLR.
