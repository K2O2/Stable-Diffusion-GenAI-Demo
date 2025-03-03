from pathlib import Path
import gradio as gr
import numpy as np
from tqdm.auto import tqdm
import openvino as ov
import openvino_genai as ov_genai
import re
import datetime
from PIL import Image
from functools import lru_cache
from typing import Tuple
from contextlib import contextmanager

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 832

MODEL_PATH = Path("models/openvino")
LORA_PATH = Path("models/lora")
OUTPUT_DIR = Path("output")

# 自动创建目录（使用mkdir替代makedirs）
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_lora_file_list() -> list[str]:
    return [f.name for f in LORA_PATH.iterdir() if f.is_file()]


def get_module_dir_list() -> list[str]:
    return [d.name for d in MODEL_PATH.iterdir() if d.is_dir()]


def device_widget(
    default: str = "GPU", exclude: list[str] = None
) -> tuple[list[str], str]:
    core = ov.Core()
    supported_devices = core.available_devices

    exclude = exclude or []
    return (
        [d for d in supported_devices if d not in exclude],
        default if default in supported_devices else supported_devices[0],
    )


def prepare_adapter_config(
    lora_list: list[Tuple[str, float]]
) -> ov_genai.AdapterConfig:
    adapter_config = ov_genai.AdapterConfig()
    for model_name, alpha in lora_list:
        lora_path = LORA_PATH / model_name
        if not lora_path.exists():
            raise ValueError(f"LoRA模型不存在: {lora_path}")
        adapter_config.add(ov_genai.Adapter(str(lora_path)), float(alpha))
    return adapter_config


def sanitize_filename(prompt):
    """处理prompt生成安全文件名"""
    # 移除特殊字符
    safe_name = re.sub(r'[<>:"/\\|?*]', "", prompt)
    # 替换空格为下划线
    safe_name = safe_name.replace(" ", "_")
    # 合并连续下划线并截断长度
    safe_name = re.sub(r"_+", "_", safe_name)[:100]
    return safe_name or "untitled"


def save_image(image: np.ndarray, prompt: str, seed: int) -> str:
    try:
        safe_name = sanitize_filename(prompt)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = OUTPUT_DIR / f"{safe_name}_{seed}_{timestamp}.png"

        Image.fromarray(image).save(output_path)
        return str(output_path)
    except Exception as e:
        gr.Error(f"保存图片失败: {str(e)}")
        raise


@lru_cache(maxsize=2)
def load_pipeline(model_dir: Path, device: str, adapters: ov_genai.AdapterConfig):
    return ov_genai.Text2ImagePipeline(str(model_dir), device, adapters=adapters)


@contextmanager
def get_pipeline(model: str, device: str, adapters):
    model_dir = MODEL_PATH / model
    if not model_dir.exists():
        raise ValueError(f"模型不存在: {model_dir}")

    pipe = load_pipeline(model_dir, device, adapters)
    try:
        yield pipe
    finally:
        del pipe  # 显式释放资源


def infer(
    model: str,
    device: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    num_inference_steps: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[np.ndarray, int]:
    """图像生成主函数"""
    try:
        # 参数验证
        if randomize_seed:
            seed = np.random.randint(0, MAX_SEED)
        # if not (256 <= width <= MAX_IMAGE_SIZE) or not (
        #     256 <= height <= MAX_IMAGE_SIZE
        # ):
        #     raise ValueError("非法分辨率参数")

        # 准备适配器
        adapters = prepare_adapter_config(lora_list)

        # 使用上下文管理器管理pipeline
        with get_pipeline(model, device, adapters) as pipe:
            generator = ov_genai.TorchGenerator(seed)

            # 进度条回调
            with tqdm(total=num_inference_steps) as pbar:

                def callback(step, num_steps, latent):
                    pbar.update(1)
                    return False

                # 生成图像
                kwargs = {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "width": width,
                    "height": height,
                    "generator": generator,
                    "callback": callback,
                }
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt

                image_tensor = pipe.generate(**kwargs)
                image_array = image_tensor.data[0]

                # 保存并返回结果
                save_path = save_image(image_array, prompt, seed)
                print(f"生成结果已保存至: {save_path}")

                return image_array, seed

    except Exception as e:
        gr.Error(f"生成失败: {str(e)}")
        raise


css = """
.refresh {
    margin-left: -18px;
}
.dropdown-label {
    margin-right: -25px;
}
"""

with gr.Blocks(
    fill_height=True,
    fill_width=True,
    title="Stable Diffusion GenAI",
    css=css,
) as demo:

    initial_devices, default_drvice = device_widget("GPU", exclude=["NPU"])
    lora_list = []

    with gr.Row(equal_height=True):
        gr.Markdown("## Stable Diffusion OpenVINO GenAI", line_breaks=False)
        gr.Textbox(
            "OpenVINO Model:",
            text_align="center",
            container=False,
            lines=1,
            scale=0,
            min_width=130,
            elem_classes="dropdown-label",
        )
        model = gr.Dropdown(
            # label="OpenVINO Model:",
            choices=get_module_dir_list(),
            interactive=True,
            container=False,
            scale=0,
            min_width=300,
            # show_label=True
        )
        model_refresh = gr.Button("🔄", scale=0, min_width=40, elem_classes="refresh")
        gr.Textbox(
            "Load Device:",
            text_align="center",
            container=False,
            lines=1,
            scale=0,
            min_width=100,
            elem_classes="dropdown-label",
        )
        device = gr.Dropdown(
            # label="Device:",
            choices=initial_devices,
            value=default_drvice,
            interactive=True,
            container=False,
            scale=0,
            min_width=120,
            # show_label=False
        )
        device_refresh = gr.Button("🔄", scale=0, min_width=40, elem_classes="refresh")
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
                randomize_seed = gr.Checkbox(info="Seed", label="Randomize", value=True)
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

            with gr.Accordion("LoRA", open=False):
                with gr.Row(equal_height=True):
                    lora_model = gr.Dropdown(
                        info="LoRA Model",
                        show_label=False,
                        choices=get_lora_file_list(),
                        interactive=True,
                    )
                    lora_weight = gr.Slider(
                        info="Weight",
                        show_label=False,
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.01,
                    )
                    add_lora = gr.Button("Add", scale=0)
                    delete_lora = gr.Button("Delete", scale=0)

                with gr.Row():
                    lora_list_show = gr.DataFrame(
                        headers=["Model", "Weight"],
                        datatype=["str", "float"],
                        value=lora_list,
                        interactive=False,
                    )

            with gr.Accordion("Experiment Setting(Unstable)", open=False):
                with gr.Row():
                    unlock_res = gr.Checkbox(label="UnLock MAX Resolution", value=False)
                    unlock_npu = gr.Checkbox(label="UnLock NPU Device", value=False)

        with gr.Column(scale=1):
            with gr.Row():
                result = gr.Image(label="Result", interactive=False)

            with gr.Row():
                run_button = gr.Button("Run")

    def unlock_resolution(value):
        if value:
            return gr.Slider(maximum=9999), gr.Slider(maximum=9999)
        else:
            return gr.Slider(maximum=MAX_IMAGE_SIZE, value=512), gr.Slider(
                maximum=MAX_IMAGE_SIZE, value=512
            )

    unlock_res.change(unlock_resolution, inputs=unlock_res, outputs=(width, height))

    def unlock_npu_device(value):
        if value:
            new_devices, default = device_widget("NPU")
            return gr.Dropdown(choices=new_devices, value=default)
        else:
            new_devices, default = device_widget("GPU", exclude=["NPU"])
            return gr.Dropdown(choices=new_devices, value=default)

    unlock_npu.change(unlock_npu_device, inputs=unlock_npu, outputs=device)

    def update_model():
        new_list = get_module_dir_list()
        new_lora = get_lora_file_list()
        return (gr.Dropdown(choices=new_list), gr.Dropdown(choices=new_lora))

    def update_device():
        new_devices, default = device_widget("GPU", exclude=["NPU"])
        return gr.Dropdown(choices=new_devices, value=default)

    model_refresh.click(fn=update_model, outputs=(model, lora_model))
    device_refresh.click(fn=update_device, outputs=device)

    def seed_input_enable(randomize):
        return gr.Number(interactive=not randomize)

    randomize_seed.change(seed_input_enable, inputs=randomize_seed, outputs=seed)

    def add_lora_model(lora_model_value, lora_weight_value):
        # 判断参数是否正常
        if lora_model_value == "" or lora_weight_value == "":
            return lora_list
        # 判断是否已经添加过了
        for i in range(len(lora_list)):
            if lora_list[i][0] == lora_model_value:
                # 弹出info提示框
                gr.Info("The model has been added.")
                # 覆盖weight值
                lora_list[i][1] = lora_weight_value
                return lora_list
        # 添加新的lora模型
        lora_list.append([lora_model_value, lora_weight_value])
        print(lora_list)
        return lora_list

    add_lora.click(
        fn=add_lora_model, inputs=[lora_model, lora_weight], outputs=lora_list_show
    )

    def clear_lora_model(lora_model_value):
        # 判断参数是否正常
        if lora_model_value == "":
            return
        # 如果找到了，就删除
        for i in range(len(lora_list)):
            if lora_list[i][0] == lora_model_value:
                lora_list.pop(i)
                break

        return gr.DataFrame(value=lora_list)

    delete_lora.click(fn=clear_lora_model, inputs=lora_model, outputs=lora_list_show)

    gr.on(
        triggers=[run_button.click, prompt.submit, negative_prompt.submit],
        fn=infer,
        inputs=[
            # basic config
            model,
            device,
            # generation config
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            width,
            height,
            num_inference_steps,
        ],
        outputs=[result, seed],
    )

if __name__ == "__main__":
    demo.launch()
