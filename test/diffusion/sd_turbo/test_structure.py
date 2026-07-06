from sd_turbo.image_to_text import initialize_pipeline
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import BasicTransformerBlock

unet = initialize_pipeline().unet
c = unet.config
print("channels :", c.block_out_channels)     # (320, 640, 1280, 1280)
print("down     :", c.down_block_types)
print("up       :", c.up_block_types)
print("layers/blk:", c.layers_per_block, " x-attn dim:", c.cross_attention_dim)

cnt = lambda m: (sum(isinstance(x, ResnetBlock2D) for x in m.modules()),
                 sum(isinstance(x, BasicTransformerBlock) for x in m.modules()))
for i, b in enumerate(unet.down_blocks): print(f"down[{i}] {type(b).__name__:24} R,T = {cnt(b)}")
print(f"mid      {type(unet.mid_block).__name__:24} R,T = {cnt(unet.mid_block)}")
for i, b in enumerate(unet.up_blocks):   print(f"up[{i}]   {type(b).__name__:24} R,T = {cnt(b)}")
print("TOTAL R,T =", cnt(unet))   # expect (22, 16)
