# Temporal-Relational Cross-Transformers (TRX)


This repo contains code for the method introduced in the paper:

[Temporal-Relational CrossTransformers for Few-Shot Action Recognition](https://arxiv.org/)

We provide two ways to use this method. The first is to incorporate it into your own few-shot video framework to allow direct comparisons against your method using the same codebase. This is recommended, as everyone has different systems, data storage etc. The second is a full train/test framework, which you will need to modify to suit your system.


## Use within your own few-shot framework (recommended)

TRX_CNN in model.py contains a TRX with multiple cardinalities (i.e. pairs, triples etc.) and a ResNet backbone. It takes in support set videos, support set labels and query videos. It outputs the distances from each query video to each of the query-specific support set prototypes which are used as logits. Feed this into the loss from utils.py. An example of how it is constructed with the required arguments, and how it is called (with input dimensions etc.) is in main in model.py

You can use it with ResNet18 with 84x84 resolution on one GPU, but we recommend distributing the CNN over multiple GPUs so you can use ResNet50, 224x224 and 5 query videos per class. How you do this will depend on your system, but the function distribute shows how we do it.

Use episodic training. That is, construct a random task from the training dataset like e.g. MAML, prototypical nets etc.. Average gradients and backpropogate once every 16 training tasks. You can look at the rest of the code for an example of how this is done.



## Use with our framework

It includes the training and testing process, data loader, logging and so on. It's fairly system specific, in particular the data loader, so it is recommended that you use within your own framework (see above).

Download your chosen dataset, and extract frames to be of the form dataset/class/video/frame-number.jpg (8 digits, zero-padded).
To prepare your data, zip the dataset folder with no compression. We did this as our filesystem has a large block size and limited number of individual files, which means one large zip file has to be stored in RAM. If you don't have this limitation (hopefully you won't because it's annoying) then you may prefer to use a different data loading process.

Put your desired splits (we used https://github.com/ffmpbgrnn/CMN for Kinetics and SSv2) in text files. These should be called trainlistXX.txt and testlistXX.txt. XX is a 0-padded number, e.g. 01. You can have separate text files for evaluating on the validation set, e.g. trainlist01.txt/testlist01.txt to train on the train set and evaluate on the the test set, and trainlist02.txt/testlist02.txt to train on the train set and evaluate on the validation set. The number is passed as a command line argument.

Modify the distribute function in model.py. We have 4 x 11GB GPUs, so we split the ResNets over the 4 GPUs and leave the cross-transformer part on GPU 0. The ResNets are always split evenly across all GPUs specified, so you might have to split the cross-transformer part, or have the cross-transformer part on its own GPU.

Modify the command line parser in run.py so it has the correct paths and filenames for the dataset zip and split text files.


## Acknowledgements

We based our code on [CNAPs](https://github.com/cambridge-mlg/cnaps) (logging, training, evaluation etc.). We use [torch_videovision](https://github.com/hassony2/torch_videovision) for video transforms. We took inspiration from the image-based [CrossTransformer](https://proceedings.neurips.cc/paper/2020/file/fa28c6cdf8dd6f41a657c3d7caa5c709-Paper.pdf) and the [Temporal-Relational Network](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.pdf).
