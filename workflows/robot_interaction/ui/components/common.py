"""Common UI widgets shared across components"""

import gradio as gr


def create_collapsible_section(title: str, content_fn, open: bool = False, visible: bool = True):
    """Create a collapsible accordion section"""
    with gr.Accordion(title, open=open, visible=visible) as accordion:
        content_fn()
    return accordion


def create_file_picker(label: str, placeholder: str = "", **kwargs):
    """Create a file/directory picker textbox"""
    return gr.Textbox(
        label=label,
        placeholder=placeholder,
        **kwargs
    )


def create_url_input(label: str, value: str = "", **kwargs):
    """Create a URL input field"""
    return gr.Textbox(
        label=label,
        value=value,
        placeholder="http://192.168.10.123:8010",
        **kwargs
    )


def create_device_selector(value: str = "cuda", **kwargs):
    """Create a device selection radio"""
    return gr.Radio(
        choices=["cuda", "cpu"],
        value=value,
        label="Device",
        **kwargs
    )


def create_fps_slider(value: float = 30.0, **kwargs):
    """Create an FPS slider"""
    return gr.Slider(
        minimum=1,
        maximum=60,
        value=value,
        step=1,
        label="FPS (Hz)",
        **kwargs
    )


def create_strategy_dropdown(value: str = "base", **kwargs):
    """Create a strategy selection dropdown with descriptions"""
    return gr.Dropdown(
        choices=[
            "base",
            "sentry",
            "highlight",
            "dagger",
            "episodic"
        ],
        value=value,
        label="Recording Strategy",
        info="base: inference only | sentry: always record | highlight: save on trigger | dagger: human intervention | episodic: episode-based recording",
        **kwargs
    )


def create_inference_type_selector(value: str = "sync", **kwargs):
    """Create inference type radio selector"""
    return gr.Radio(
        choices=["sync", "rtc", "chunk"],
        value=value,
        label="Inference Type",
        info=(
            "sync: full chunk execution | "
            "rtc: Real-Time Chunking for faster response | "
            "chunk: open-loop (infer every chunk_interval_s, send n_action_steps)"
        ),
        **kwargs
    )
