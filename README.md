# ACN
Using all convoluntional network to train and test on CIFAR10 and CIFAR100

This is an re-implement of the paper ["STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET"](https://arxiv.org/abs/1412.6806).

I have implemented two of basic network architectures in this ACN paper with Keras: ConvPool-CNN-C and All-CNN-C.

# 1.ConvPool-CNN-C
I have made all operations described in this paper except ZCA whitening, which is found not to help the metwork converge. This code file is conPool-CNN.py.

On CIFAR10 dataset, this model get 90.03% acc after 2000 epochs. Here is the curve for training processing.![image](https://github.com/OrangeCat95/ACN/blob/master/pic/conv-poolc.bmp)

On CIFAR100 dataset, this model get 61.99% acc after 1500 epochs.![image](https://github.com/OrangeCat95/ACN/blob/master/pic/2.bmp)

# 2.All-Conv-C
I have made all operations described in this paper except ZCA whitening.  The code file is allconv_cifar10.py and allconv_cifar100.py.

On CIFAR10 dataset, this model get 91.91% after 3000 epochs (test best in 2993).![image](https://github.com/OrangeCat95/ACN/tree/master/pic/acn1.png)

On CIFAR100 dataset, this model get 66.13% after 3500 epochs (test best in 3229).![image](https://github.com/OrangeCat95/ACN/tree/master/pic/acn2.png)
