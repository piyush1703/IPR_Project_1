# Implementation Details

## Installation
* Clone this repo into a directory
* Organize datasets as required
* Install python dependencies.
* Python version 3.9.13 is required
* torch version 1.8.0+cpu and torch vision version 0.9.0+cpu
* if encountering any error while downloading torch and torchvision version use following command
```
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
* Install requirements by following command
```
pip install -r requirements.txt
```
## Datasets
* Dataset downloaded from the paper repository are in folder **part_A_final** and **part_B_final**
* These datasets are preprocessed using `preprocess_dataset1.py` , `preprocess_dataset2.py` and `preprocess_dataset3.py` scripts <br> to make it in form required as mentioned below
* Preprocessed Datasets are present in **DATA_ROOT (SHTechPartA)** and **DATA_ROOT2 (SHTechPartB)**

## Organize the counting dataset
We use a list file to collect all the images and their ground truth annotations in a counting dataset. When your dataset is organized as recommended in the following, the format of this list file is defined as:
```
train/scene01/img01.jpg train/scene01/img01.txt
train/scene01/img02.jpg train/scene01/img02.txt
...
```

### Dataset structures:
```
DATA_ROOT/
        |->train/
        |    |->scene01/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->...
        |->train.list
        |->test.list
```

### Annotations format
For the annotations of each image, we use a single txt file which contains one annotation per line. Note that indexing for pixel values starts at 0. The expected format of each line is:
```
x1 y1
x2 y2
...
```

## Training

* The network can be trained using the `train.py` script 
* For training on SHTechPartA use DATA_ROOT in place of $DATA_ROOT in below command 
* For training on SHTechPartB use DATA_ROOT2 in place of $DATA_ROOT in below command

```
    python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
```
By default, a periodic evaluation will be conducted on the validation set.

## Testing

Store trained model weights in "./weights" folder, run the following commands to launch a visualization demo:

```
python run_test.py --weight_path ./weights/your_weights_file --output_dir ./logs/
```
Visualised image can be seen in logs directory
