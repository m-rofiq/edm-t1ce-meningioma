# from models.unet_baseline import UNetBaseline
# from models.resunet import ResUNet
# from models.attention_unet import AttentionUNet
# from models.resattention_unet import ResAttentionUNet
# from models.dense_unet import DenseUNet
# from models.dmec_net_step1 import DMECNetStep1
# from models.dmec_net_step2 import DMECNetStep2
# from models.dmec_net_step3 import DMECNetStep3
# from models.guided_resattention_unet import GuidedResAttentionUNet
# from models.edm_synth import EDMSynth
# from models.discriminators.patchgan import PatchGANDiscriminator
from ddresunet_model import DDResUNet


MODEL_REGISTRY = {

    # "UNetBaseline": UNetBaseline,
    # "ResUNet": ResUNet,
    # "AttentionUNet": AttentionUNet,
    # "ResAttentionUNet": ResAttentionUNet,
    # "DenseUNet": DenseUNet,
    # "DMECNetStep1": DMECNetStep1,
    # "DMECNetStep2" : DMECNetStep2,
    # "DMECNetStep3" : DMECNetStep3,
    # "GuidedResAttentionUNet": GuidedResAttentionUNet,
    # "EDMSynth" : EDMSynth,
    # "patchgan" : PatchGANDiscriminator,
    "DDResUNet": DDResUNet
}