import os
import sys
import gradio as gr
import numpy as np
from tqdm.auto import tqdm
import openvino as ov
import openvino_genai as ov_genai

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 832
MODEL_PATH = "models/openvino"  # the path of the module folder
LORA_PATH = "models/lora"  # the path of the lora folder

css = """
.refresh {
    margin-left: -18px;
}
.dropdown-label {
    margin-right: -25px;
}
"""


def prepare_adapter_config(adapters):
    adapter_config = ov_genai.AdapterConfig()

    # Multiple LoRA adapters applied simultaneously are supported, parse them all and corresponding alphas from cmd parameters:
    for i in range(int(len(adapters) / 2)):
        adapter = ov_genai.Adapter(adapters[2 * i])
        alpha = float(adapters[2 * i + 1])
        adapter_config.add(adapter, alpha)

    return adapter_config


def get_lora_file_list():
    return [
        f for f in os.listdir(LORA_PATH) if os.path.isfile(os.path.join(LORA_PATH, f))
    ]  # return the file name in lora folder


def device_widget(default="CPU", exclude=None, added=None):
    core = ov.Core()

    supported_devices = core.available_devices

    exclude = exclude or []
    for ex_device in exclude:
        if ex_device in supported_devices:
            supported_devices.remove(ex_device)

    added = added or []
    for add_device in added:
        if add_device not in supported_devices:
            supported_devices.append(add_device)

    return supported_devices, default


# get the path for modules in the module folder,layer for once,list the dir name in module folder
def get_module_dir_list():
    return [
        d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))
    ]  # return the dir name in module folder


def make_demo():
    def infer(
        model_dir,
        device,
        prompt,
        negative_prompt,
        seed,
        randomize_seed,
        width,
        height,
        num_inference_steps,
        # progress=gr.Progress(track_tqdm=True),
    ):
        model_dir = os.path.join(MODEL_PATH, model_dir)

        pipe = ov_genai.Text2ImagePipeline(
            model_dir, device, adapters=ov_genai.AdapterConfig()
        )

        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)

        generator = ov_genai.TorchGenerator(seed)

        pbar = tqdm(total=num_inference_steps)

        def callback(step, num_steps, latent):
            pbar.update(1)
            sys.stdout.flush()
            return False

        # if negative_prompt is none then do not transfer negative prompt
        # because some models do not support negative prompt
        if negative_prompt == "":
            image_tensor = pipe.generate(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator,
                adapters=ov_genai.AdapterConfig(),
                callback=callback,
            )
        else:
            image_tensor = pipe.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator,
                adapters=ov_genai.AdapterConfig(),
                callback=callback,
            )

        return image_tensor.data[0], seed

    with gr.Blocks(
        fill_height=True,
        fill_width=True,
        title="Stable Diffusion GenAI",
        css=css,
    ) as demo:
        with gr.Row(equal_height=True):
            gr.Markdown("## Stable Diffusion OpenVINO GenAI", line_breaks=False)
            model_dir = gr.Textbox(
                "OpenVINO Model:",
                text_align="center",
                container=False,
                lines=1,
                scale=0,
                min_width=130,
                elem_classes="dropdown-label",
            )
            model_dropdown = gr.Dropdown(
                # label="OpenVINO Model:",
                choices=get_module_dir_list(),
                interactive=True,
                container=False,
                scale=0,
                min_width=300,
                # show_label=True
            )
            model_refresh = gr.Button(
                "🔄", scale=0, min_width=40, elem_classes="refresh"
            )

            device = gr.Textbox(
                "Load Device:",
                text_align="center",
                container=False,
                lines=1,
                scale=0,
                min_width=100,
                elem_classes="dropdown-label",
            )
            initial_devices, default_drvice = device_widget("GPU", exclude=["NPU"])
            device_dropdown = gr.Dropdown(
                # label="Device:",
                choices=initial_devices,
                value=default_drvice,
                interactive=True,
                container=False,
                scale=0,
                min_width=120,
                # show_label=False
            )
            device_refresh = gr.Button(
                "🔄", scale=0, min_width=40, elem_classes="refresh"
            )
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    # prompt
                    prompt = gr.Textbox(
                        info="Prompt:",
                        show_label=False,
                        placeholder="Type a prompt here...",
                        lines=3,
                        interactive=True,
                    )
                with gr.Row():
                    # negative prompt
                    negative_prompt = gr.Textbox(
                        info="Negative Prompt:",
                        show_label=False,
                        placeholder="Attention!!! Some models do not support negative prompt.Please leave it blank.",
                        lines=1,
                        interactive=True,
                    )

                with gr.Row():
                    num_inference_steps = gr.Slider(
                        info="Steps",
                        show_label=False,
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                    )

                with gr.Row():
                    width = gr.Slider(
                        info="Width",
                        show_label=False,
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=64,
                        value=512,
                    )

                    height = gr.Slider(
                        info="Height",
                        show_label=False,
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=64,
                        value=512,
                    )

                with gr.Row(equal_height=True):
                    randomize_seed = gr.Checkbox(
                        info="Seed", label="Randomize", value=True
                    )
                    seed = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        show_label=False,
                        precision=0,
                        interactive=False,
                        scale=10,
                    )

                # with gr.Accordion("LoRA", open=False):
                #     with gr.Row(equal_height=True):
                #         lora_model = gr.Dropdown(
                #             info="LoRA Model",
                #             show_label=False,
                #             choices=utils.get_lora_file_list(),
                #             interactive=True,
                #         )
                #         lora_weight = gr.Slider(
                #             info="Weight",
                #             show_label=False,
                #             minimum=0,
                #             maximum=1,
                #             step=0.1,
                #             value=0.5,
                #         )
                #         add_lora = gr.Button("Add", scale=0)

                #     lora_list = []
                #     with gr.Row():
                #         lora_list_show = gr.Dataframe(
                #             show_label=False,
                #             headers=["Model", "Weight"],
                #             datatype=["str", "float"],
                #             interactive=True,
                #             wrap=True,
                #         )

            with gr.Column(scale=1):
                with gr.Row():
                    result = gr.Image(label="Result", interactive=False)

                with gr.Row():
                    run_button = gr.Button("Run")

        # def add_lora_model():
        #     lora_list.append(lora_model, lora_weight)
        #     return lora_list

        # add_lora.click(add_lora_model, outputs=lora_list)
        def update_model():
            new_list = get_module_dir_list()
            return gr.Dropdown(choices=new_list)

        def update_device():
            new_devices, default = device_widget("GPU", exclude=["NPU"])
            return gr.Dropdown(choices=new_devices, value=default)

        model_refresh.click(fn=update_model, outputs=model_dropdown)
        device_refresh.click(fn=update_device, outputs=device_dropdown)

        def seed_input_enable(randomize):
            return gr.Number(interactive=not randomize)

        randomize_seed.change(seed_input_enable, inputs=randomize_seed, outputs=seed)

        gr.on(
            triggers=[run_button.click, prompt.submit, negative_prompt.submit],
            fn=infer,
            inputs=[
                model_dropdown,
                device_dropdown,
                prompt,
                negative_prompt,
                seed,
                randomize_seed,
                width,
                height,
                num_inference_steps,
            ],
            outputs=[result,seed],
        )

    return demo


demo = make_demo()
if __name__ == "__main__":
    demo.launch()
