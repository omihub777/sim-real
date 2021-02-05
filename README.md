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
* Cosine LR Scheduler
* Mixed Precision
* Early Stopping
* Size argument for resize


## Done
* GPU-mode: TF2 automatically run on gpu if available
* Run on Colab: git clone
