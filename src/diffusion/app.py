import random
import gradio as gr
from sd_turbo.image_to_text import generation, initialize_pipeline
from sd_turbo_fused.resnet.fused_resnet_block import patch_unet
from sd_turbo_fused.transformer.fused_ffn_block import patch_unet_ffn

_pipes = {}


def get_pipeline(use_fused: bool):
    if use_fused not in _pipes:
        _pipes[use_fused] = initialize_pipeline()
        if use_fused:
            patch_unet(_pipes[use_fused].unet, verbose=False)
            patch_unet_ffn(_pipes[use_fused].unet, verbose=False)
    return _pipes[use_fused]


def generate_image(prompt: str, seed: int, use_fused: bool):
    pipe = get_pipeline(use_fused)
    image = generation(pipeline=pipe, prompt=prompt, seed=seed, out_dir="diffusion/outputs")
    return image


with gr.Blocks(title="MLA Project: SD-Turbo Image Generator with Fused CuTile Kernels") as demo:
    gr.Markdown("## MLA Project: SD-Turbo Image Generator with Fused CuTile Kernels")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter a text prompt…",
                value="Will Smith eating spaghetti",
                lines=3,
            )
            with gr.Row():
                seed = gr.Number(label="Seed", value=0, precision=0, minimum=0, maximum=2**31 - 1)
                with gr.Column(min_width=80):
                    randomize_btn = gr.Button("🎲")
                    use_fused = gr.Checkbox(label="Use fused kernels", value=True)
            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    run_btn.click(
        fn=generate_image,
        inputs=[prompt, seed, use_fused],
        outputs=output_image,
    )

    randomize_btn.click(
        fn=lambda: random.randint(0, 2**31 - 1),
        inputs=[],
        outputs=seed,
    )


if __name__ == "__main__":
    demo.launch()
