"""Camera panel UI components"""

import gradio as gr


def create_camera_panel():
    """Create camera preview configuration panel"""
    components = {}

    with gr.Accordion("Camera Settings", open=True) as panel:
        components["camera_list"] = gr.CheckboxGroup(
            choices=["right_eye", "left_eye", "left_wrist", "right_wrist"],
            value=["right_eye", "left_eye", "left_wrist", "right_wrist"],
            label="Cameras to Show",
            info="Select which cameras to display"
        )

        components["fps"] = gr.Slider(
            minimum=1,
            maximum=60,
            value=30,
            step=1,
            label="Display FPS"
        )

        components["show_quad"] = gr.Checkbox(
            label="Quad Layout",
            value=True,
            info="Show all 4 cameras in 2x2 grid"
        )

        with gr.Row():
            components["window_width"] = gr.Number(
                label="Window Width",
                value=640,
                precision=0,
                minimum=320
            )
            components["window_height"] = gr.Number(
                label="Window Height",
                value=480,
                precision=0,
                minimum=240
            )

    components["panel"] = panel
    return components
