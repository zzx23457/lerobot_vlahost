"""Replay panel UI components"""

import gradio as gr
from .common import create_fps_slider


def create_replay_dataset_panel():
    """Create dataset configuration panel for replay mode"""
    components = {}

    with gr.Accordion("Dataset Settings", open=True) as panel:
        components["repo_id"] = gr.Textbox(
            label="HuggingFace Repo ID",
            placeholder="username/dataset_name",
            info="Required: dataset to replay from"
        )

        components["episode"] = gr.Number(
            label="Episode Index",
            value=0,
            precision=0,
            minimum=0,
            info="Which episode to replay"
        )

        components["dataset_root"] = gr.Textbox(
            label="Local Dataset Root (optional)",
            placeholder="datasets/",
            info="Local cache directory to avoid re-downloading"
        )

        components["fps"] = create_fps_slider(value=30.0)

    components["panel"] = panel
    return components
