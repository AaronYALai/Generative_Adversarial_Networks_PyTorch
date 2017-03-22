Generative Adversarial Networks in PyTorch
=======


[![Build Status](https://travis-ci.org/AaronYALai/Generative_Adversarial_Networks_PyTorch.svg?branch=master)](https://travis-ci.org/AaronYALai/Generative_Adversarial_Networks_PyTorch)
[![Coverage Status](https://coveralls.io/repos/github/AaronYALai/Generative_Adversarial_Networks_PyTorch/badge.svg?branch=master)](https://coveralls.io/github/AaronYALai/Generative_Adversarial_Networks_PyTorch?branch=master)

About
--------

The repo is about the implementations of GAN, DCGAN, Improved GAN, LAPGAN, and InfoGAN in PyTorch.

My presentation about GANs' recent development: [Presentation slides](https://docs.google.com/presentation/d/1HRNjCo_0PlspynoJKuoEF1AYkaKaUNgMzQ4nqiTlNUM/edit#slide=id.p)

Presented in the group meeting of Machine Discovery and Social Network Mining Lab, National Taiwan University.

Content
--------

- GAN: 

- DC-GAN: 

- LAP-GAN: 

- Improved GAN: 

- Info-GAN: 


Usage
--------
Clone the repo and use the [virtualenv](http://www.virtualenv.org/):

    git clone https://github.com/AaronYALai/Generative_Adversarial_Networks_PyTorch

    cd Generative_Adversarial_Networks_PyTorch

    virtualenv venv

    source venv/bin/activate

Install pytorch and all dependencies and run the model (in Linux):

    pip install https://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp27-none-linux_x86_64.whl 

    pip install torchvision

    pip install -r requirements.txt

    cd GAN

    python run_GAN.py

More details about the installation about PyTorch: <http://pytorch.org>


References
--------

- GAN: I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” NIPS, 2014.

- DC-GAN: Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv 2015.

- LAP-GAN: Denton, Emily L., Soumith Chintala, and Rob Fergus. "Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks." NIPS 2015.

- Improved GAN: Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. “Improved techniques for training gans.” NIPS 2016.

- Info-GAN: Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., & Abbeel, P. “Infogan: Interpretable representation learning by information maximizing generative adversarial nets.” NIPS 2016.
