from unittest import TestCase
from GAN.run_GAN import run_GAN
from DCGAN.run_DCGAN import run_DCGAN


class Test_running(TestCase):

    def test_run_GAN(self):
        run_GAN(n_epoch=1, batch_size=100, n_update_dis=5, n_update_gen=1,
                update_max=50)

    def test_run_DCGAN(self):
        run_DCGAN(n_epoch=1, batch_size=10, n_update_dis=1, n_update_gen=1,
                  update_max=10)
