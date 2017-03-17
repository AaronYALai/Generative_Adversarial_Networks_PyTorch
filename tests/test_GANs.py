from unittest import TestCase
from GAN.run_GAN import run_GAN


class Test_running(TestCase):

    def test_run_GAN(self):
        run_GAN(n_epoch=1, batch_size=100, n_update_dis=5, n_update_gen=1)
