Generative Adversarial Networks in PyTorch
=======


[![Build Status](https://travis-ci.org/AaronYALai/Generative_Adversarial_Networks_PyTorch.svg?branch=master)](https://travis-ci.org/AaronYALai/Generative_Adversarial_Networks_PyTorch)
[![Coverage Status](https://coveralls.io/repos/github/AaronYALai/Generative_Adversarial_Networks_PyTorch/badge.svg?branch=master)](https://coveralls.io/github/AaronYALai/Generative_Adversarial_Networks_PyTorch?branch=master)

About
--------

Implementations about GAN, Improved GAN, DCGAN, LAPGAN, and InfoGAN in PyTorch

GANs' recent development presented in the group meeting of MS Lab at National Taiwan University: [Slides](https://docs.google.com/presentation/d/1HRNjCo_0PlspynoJKuoEF1AYkaKaUNgMzQ4nqiTlNUM/edit#slide=id.p)



Content
--------


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
