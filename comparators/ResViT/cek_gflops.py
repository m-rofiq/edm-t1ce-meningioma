from ptflops import get_model_complexity_info
g = model.netG.module if hasattr(model.netG, "module") else model.netG
g.eval()
macs, params = get_model_complexity_info(
    g, (3, 256, 256), as_strings=False, print_per_layer_stat=False)
print(f"GFLOPs {2*macs/1e9:.2f}   params {params:,d}")