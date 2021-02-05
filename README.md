# Train Image Classifier using Synthetic Data

Train NNs using only synthetic images.
Test NNs using real images.
## Data
* Directories:
    * **mask**: Synthetic Images(ONLY objects) **USED FOR TRAINING**.
    * **trainB**: Real Images for training(objects+tray). NOT USED.
    * **valB**: Real Images for validation(objects+tray). MIGHT BE USED FOR TEST.
    * **valB_crop**: Real Images for validation(ONLY objects). **USED FOR TEST**.



## TODO
* create valB_crop
* Save weights
* Log results
* Other models
* Early Stopping
* Size argument for resize


## Done
* GPU-mode: TF2 automatically run on gpu if available
* Run on Colab: git clone
* Mixed Precision: P100 is not compatible with Mixed Precision...
* Cosine LR Scheduler
* Warmup

## Ref.
* [Building an image data pipeline](https://cs230.stanford.edu/blog/datapipeline/#building-an-image-data-pipeline)
Custom Datset in TF
* [Mixed precision](https://www.tensorflow.org/guide/mixed_precision)
* 