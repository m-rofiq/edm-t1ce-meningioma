from models.pgan_networks import PGANGenerator, PGANDiscriminator
 
MODEL_REGISTRY = {
    "pGAN": PGANGenerator,
    "pGAN_D": PGANDiscriminator,
 
    # Tambahkan di sini kalau nanti Anda implementasikan cGAN (Opsi A/B di
    # strategi_adaptasi_pGAN_cGAN.md) di folder proyek yang sama, mis.:
    # "cGAN_G_A": CycleGANGeneratorAtoB,
    # "cGAN_G_B": CycleGANGeneratorBtoA,
}