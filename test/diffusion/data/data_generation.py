import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.resnet import ResnetBlock2D
from sd_turbo.image_to_text import initialize_pipeline
from load_helper import data_path

# Global variables
OUT_DIR = data_path()
SEED = 0


def norm_silu(pipe, prompt):
    """
    Captures a ground truth block of the SD-Turbo model.

    Args:
        pipe (_type_): SD-Turbo pipeline (blocks)
        prompt (_type_): given prompt

    Raises:
        RuntimeError: Ensures the forward hook ran
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    block = next(m for m in pipe.unet.modules() if isinstance(m, ResnetBlock2D) and m.norm1.num_channels == 320)
    gn = block.norm1
    print(f"target GroupNorm: groups={gn.num_groups} channels={gn.num_channels} eps={gn.eps}")

    captured = {}

    def hook(module, inputs, output):
        # Fire only once: capture the first U-Net forward, then we remove the hook.
        if captured:
            return
        x = inputs[0]      # input to GroupNorm
        gn_out = output    # real GroupNorm(x)
        y = F.silu(gn_out) # the fused chain's target
        captured["x"]      = x.detach().to(torch.float16).cpu().numpy()
        captured["weight"] = module.weight.detach().to(torch.float16).cpu().numpy()
        captured["bias"]   = module.bias.detach().to(torch.float16).cpu().numpy()
        captured["y"]      = y.detach().to(torch.float16).cpu().numpy()

    # Capture input / output of group norm
    handle = gn.register_forward_hook(hook)

    # Make generation reproducable
    generator = torch.Generator(device=device).manual_seed(SEED)
    _ = pipe(prompt, num_inference_steps=1, guidance_scale=0.0, generator=generator)
    handle.remove()

    if not captured:
        raise RuntimeError("hook never fired -- no matching ResnetBlock ran")

    # 3) Save results - *.npz
    np.savez(
        OUT_DIR,
        groups=np.int64(gn.num_groups),
        num_channels=np.int64(gn.num_channels),
        eps=np.float32(gn.eps),
        **captured,
    )
    print(f"wrote {OUT_DIR}")
    print(f"  x{captured['x'].shape}  y{captured['y'].shape}  "
          f"weight{captured['weight'].shape}  bias{captured['bias'].shape}")
    print(f"  x range [{captured['x'].min():.3f}, {captured['x'].max():.3f}]  "
          f"y range [{captured['y'].min():.3f}, {captured['y'].max():.3f}]")
    return


if __name__ == "__main__":
    pipe = initialize_pipeline()
    prompt = "A Star Wars Storm Trooper holding a light saber"
    norm_silu(pipe, prompt)
