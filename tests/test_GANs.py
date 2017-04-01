from unittest import TestCase
from GAN.run_GAN import run_GAN
from DCGAN.run_DCGAN import run_DCGAN
from ImprovedGAN.run_ImprovedGAN import run_ImprovedGAN
from LAPGAN.run_LAPGAN import run_LAPGAN
from InfoGAN.run_InfoGAN import run_InfoGAN


class Test_running(TestCase):

    def test_run_GAN(self):
        run_GAN(n_epoch=1, batch_size=20, n_update_dis=5, n_update_gen=1,
                update_max=20)

    def test_run_DCGAN(self):
        run_DCGAN(n_epoch=1, batch_size=10, n_update_dis=1, n_update_gen=1,
                  update_max=20)

    def test_run_ImprovedGAN(self):
        run_ImprovedGAN(n_epoch=1, batch_size=10, n_update_dis=1,
                        n_update_gen=1, update_max=20)

    def test_run_LAPGAN(self):
        run_LAPGAN(n_epoch=1, batch_size=10, n_update_dis=1,
                   n_update_gen=1, update_max=20)

    def test_run_InfoGAN(self):
        run_InfoGAN(n_conti=2, n_discrete=1, D_featmap_dim=64,
                    G_featmap_dim=128, n_epoch=1, update_max=60)
