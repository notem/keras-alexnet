## Keras-Alexnet

### Usage
1) ``python alexnet.py [keras-alexnet.h5]`` - build, train, and test the model on the CIFAR-100 dataset
2) ``python gradcam.py [keras-alexnet.h5] [example.jpg]`` - apply the trained alexnet model on an example image and produce a Grad-CAM visualization of the neuron activation

### References
* [https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* [http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
* [http://image-net.org/challenges/LSVRC/2012/supervision.pdf](http://image-net.org/challenges/LSVRC/2012/supervision.pdf)
* [https://keras.io/](https://keras.io/)
* [http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)