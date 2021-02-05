# Train Image Classifier using Synthetic Data

Train NNs using only synthetic images.
Test NNs using real images.

## TODO(High)
### High
* create valB_crop
* Other models
* Size argument for resize
* Log learning rate

### Low
* Oputuna(for lr, warmup_lr, batch_size, ...)


## Done
### High
* GPU-mode: TF2 automatically run on gpu if available
* Run on Colab: git clone
* Mixed Precision: P100 is not compatible with Mixed Precision...
* Cosine LR Scheduler
* Warmup
* Log results: Need [comet.ml](https://www.comet.ml/) api-key in data/api_key.txt
* Check with cifar-10/mnist
* Early Stopping
* Save weights


### Low



## Ref.
* [Building an image data pipeline](https://cs230.stanford.edu/blog/datapipeline/#building-an-image-data-pipeline)
Custom Datset in TF
* [Mixed precision](https://www.tensorflow.org/guide/mixed_precision)
* 