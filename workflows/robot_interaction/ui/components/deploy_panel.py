"""Deploy panel UI components"""

import gradio as gr
from .common import (
    create_file_picker,
    create_device_selector,
    create_fps_slider,
    create_strategy_dropdown,
    create_inference_type_selector,
)


def create_deploy_policy_panel():
    """Create policy configuration panel for deploy mode"""
    components = {}

    with gr.Accordion("Policy Settings", open=True) as panel:
        components["policy_path"] = create_file_picker(
            label="Model Path",
            placeholder="outputs/train/act_v2/pretrained_model or outputs/train/smolvla/pretrained_model"
        )
        components["policy_device"] = create_device_selector(value="cuda")

    components["panel"] = panel
    return components


def create_deploy_inference_panel():
    """Create inference configuration panel for deploy mode"""
    components = {}

    with gr.Accordion("Inference Settings", open=False) as panel:
        components["fps"] = create_fps_slider(value=30.0)
        components["strategy"] = create_strategy_dropdown(value="base")
        components["inference_type"] = create_inference_type_selector(value="sync")

        # RTC-specific settings (conditional visibility)
        with gr.Group(visible=False) as rtc_group:
            gr.Markdown("### Real-Time Chunking Settings")
            components["execution_horizon"] = gr.Slider(
                minimum=1,
                maximum=100,
                value=4,
                step=1,
                label="Execution Horizon",
                info="Number of actions to execute before re-inference"
            )
            components["max_guidance_weight"] = gr.Slider(
                minimum=0,
                maximum=20,
                value=10.0,
                step=0.5,
                label="Max Guidance Weight"
            )

        components["rtc_group"] = rtc_group

        components["duration"] = gr.Slider(
            minimum=0,
            maximum=600,
            value=0,
            step=10,
            label="Duration (seconds)",
            info="0 = infinite"
        )

        components["interpolation_multiplier"] = gr.Slider(
            minimum=1,
            maximum=10,
            value=1,
            step=1,
            label="Interpolation Multiplier"
        )

        components["use_torch_compile"] = gr.Checkbox(
            label="Use Torch Compile",
            value=False,
            info="PyTorch 2.0+ compilation for faster inference"
        )

        components["show_cameras"] = gr.Checkbox(
            label="Show Camera Windows",
            value=False
        )

        # Camera rename map editor
        with gr.Accordion("Camera Rename Map (Advanced)", open=False):
            gr.Markdown("Map physical camera names to model observation keys")
            components["rename_map_json"] = gr.Textbox(
                label="Rename Map (JSON)",
                placeholder='{"camera1": "image", "camera2": "wrist_image"}',
                lines=3,
                info="Leave empty if not needed"
            )

    components["panel"] = panel
    return components


def create_deploy_dataset_panel():
    """Create dataset configuration panel for deploy mode (optional)"""
    components = {}

    with gr.Accordion("Dataset Settings (for VLA & Recording)", open=False, visible=False) as panel:
        components["repo_id"] = gr.Textbox(
            label="HuggingFace Repo ID",
            placeholder="username/dataset_name",
            info="Required for VLA models and non-base strategies"
        )

        components["dataset_root"] = gr.Textbox(
            label="Local Dataset Root (optional)",
            placeholder="datasets/",
            info="Local cache directory"
        )

        components["single_task"] = gr.Textbox(
            label="Task Description (VLA only)",
            placeholder="pick up the red block and place it in the blue bowl",
            lines=2,
            info="Natural language task description for VLA models"
        )

    components["panel"] = panel
    return components
