"""Data processing panel factory for robot_interaction UI.

创建一个 Gradio Accordion，包含数据处理 5 种 operation 的所有控件。
返回 dict，便于 app_zh.py 解包访问组件并接入事件。
"""

from dataclasses import dataclass
import gradio as gr

from .common import create_file_picker, create_collapsible_section


@dataclass
class DataProcessingPanel:
    """打包所有数据处理相关组件。"""

    panel: gr.Accordion
    operation: gr.Radio
    dataset_path: gr.Dropdown  # 从 datasets/ 自动列出
    dataset_path_refresh_btn: gr.Button
    output_path: gr.Textbox
    # sanity
    sanity_group: gr.Group
    n_samples: gr.Number
    # clean
    clean_group: gr.Group
    dry_run: gr.Checkbox
    report_only: gr.Checkbox
    zero_threshold: gr.Number
    # merge
    merge_group: gr.Group
    source_roots_text: gr.Textbox
    merge_repo_id: gr.Textbox
    merge_video_size_mb: gr.Number
    # ts_check
    ts_check_group: gr.Group
    video_key: gr.Textbox
    tolerance_ms: gr.Number
    ts_report_output: gr.Textbox
    ts_output_format: gr.Radio
    # v2_convert
    v2_convert_group: gr.Group
    v2_variant: gr.Radio
    v2_output_root: gr.Textbox
    v2_suffix: gr.Textbox
    cam_left_eye: gr.Checkbox
    cam_right_eye: gr.Checkbox
    cam_left_wrist: gr.Checkbox
    cam_right_wrist: gr.Checkbox
    v2_dry_run: gr.Checkbox


def create_data_processing_panel() -> DataProcessingPanel:
    """Create the data processing panel and return a dataclass of all widgets.

    Visibility groups (sanity/clean/merge/ts_check/v2_convert) are returned
    but their visibility is driven by ``app_zh.py``'s ``on_data_operation_change``.
    They start visible only for the default operation (`sanity`).
    """
    # 延迟导入避免循环依赖
    from workflows.robot_interaction.ui.app_zh import _list_datasets_full
    with gr.Accordion("数据处理", open=True, visible=False) as panel:
        gr.Markdown(
            "**⚠️ 默认所有写操作是 dry-run**。实际写入前请明确关闭 dry-run。"
        )

        operation = gr.Radio(
            choices=[
                ("数据集烟测 (sanity_check)", "sanity"),
                ("清洗脏 Episode (clean_dirty_episodes)", "clean"),
                ("合并数据集 (merge_two_datasets)", "merge"),
                ("时间戳对齐检查 (check_timestamp_alignment)", "ts_check"),
                ("v2 Schema 转换 (v2_convert)", "v2_convert"),
            ],
            value="sanity",
            label="操作",
        )

        with gr.Row():
            dataset_path = gr.Dropdown(
                label="数据集路径 (从 datasets/ 自动列出)",
                choices=_list_datasets_full(),
                value=None,
                allow_custom_value=True,
                info="下拉选择或在文本框里输入完整路径",
                scale=4,
            )
            dataset_path_refresh_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)
        output_path = create_file_picker(
            label="输出路径 (clean / merge / v2_convert 用)",
            placeholder="留空表示 dry-run 或使用默认推导路径",
        )

        # ---- sanity ----
        with gr.Group(visible=True) as sanity_group:
            n_samples = gr.Number(value=5, label="抽样数 (n_samples)", precision=0, minimum=1)

        # ---- clean ----
        with gr.Group(visible=False) as clean_group:
            dry_run = gr.Checkbox(value=True, label="Dry-run (推荐默认开启)")
            report_only = gr.Checkbox(value=False, label="Report-only (仅报告, 不写盘)")
            zero_threshold = gr.Number(
                value=7, label="前 N 位全零阈值 (zero_threshold)",
                precision=0, minimum=0,
            )
            gr.Markdown(
                "实际清洗会创建新目录并重映射 episode_index；"
                "**输入与输出不能是同一路径**。"
            )

        # ---- merge ----
        with gr.Group(visible=False) as merge_group:
            gr.Markdown(
                "**源数据集路径** — 每行一个绝对路径, 至少 2 个。可以从 `datasets/` 下复制, "
                "或点击下方按钮从 `_list_datasets_full()` 中自动拼接。"
            )
            source_roots_text = gr.Textbox(
                label="源数据集路径 (每行一个, 至少 2 个)",
                placeholder="datasets/26-07-21+22+23+25-merged\ndatasets/26-07-25-01-19-04",
                lines=4,
            )
            merge_repo_id = gr.Textbox(
                label="输出 repo_id",
                placeholder="例如: 26-07-merged",
            )
            merge_video_size_mb = gr.Number(
                value=0.001, label="video_files_size_mb (越小越倾向独立 mp4)",
                precision=4, minimum=0.0001,
            )
            gr.Markdown(
                "**输出目录不能已存在** (避免误覆盖)。"
            )

        # ---- ts_check ----
        with gr.Group(visible=False) as ts_check_group:
            video_key = gr.Textbox(
                label="video_key (留空则自动检测)",
                placeholder="例如: observation.images.left_eye",
            )
            tolerance_ms = gr.Number(
                value=1.0, label="tolerance_ms",
                precision=2, minimum=0,
            )
            ts_report_output = gr.Textbox(
                label="报告输出文件 (留空表示 stdout)",
                placeholder="例如: outputs/ts_check_report.json",
            )
            ts_output_format = gr.Radio(
                choices=["text", "json"], value="text", label="输出格式",
            )
            gr.Markdown(
                "退出码: 0=干净 / 1=发现 drift (业务结果, UI 视为成功) / 其它=错误"
            )

        # ---- v2_convert ----
        with gr.Group(visible=False) as v2_convert_group:
            v2_variant = gr.Radio(
                choices=[
                    ("standard (action 取当前步)", "standard"),
                    ("next_joint (action 取下一步 joint_pos)", "next_joint"),
                ],
                value="standard",
                label="转换变体",
                info="两种转换产出的 action[:, 14:-2] 含义不同，请根据下游策略选择",
            )
            v2_output_root = gr.Textbox(
                label="v2 输出路径 (留空则按 v2_suffix 自动推导)",
                placeholder="datasets/foo_v2",
            )
            v2_suffix = gr.Textbox(value="_v2", label="v2_suffix")
            gr.Markdown("**相机保留开关 (取消勾选则丢弃该相机)**")
            with gr.Row():
                cam_left_eye = gr.Checkbox(value=True, label="left_eye")
                cam_right_eye = gr.Checkbox(value=True, label="right_eye")
                cam_left_wrist = gr.Checkbox(value=True, label="left_wrist")
                cam_right_wrist = gr.Checkbox(value=True, label="right_wrist")
            v2_dry_run = gr.Checkbox(value=True, label="Dry-run")
            gr.Markdown(
                "v1 永远不会被修改；rollback: `rm -rf <v2_dir>`"
            )

    return DataProcessingPanel(
        panel=panel,
        operation=operation,
        dataset_path=dataset_path,
        dataset_path_refresh_btn=dataset_path_refresh_btn,
        output_path=output_path,
        sanity_group=sanity_group,
        n_samples=n_samples,
        clean_group=clean_group,
        dry_run=dry_run,
        report_only=report_only,
        zero_threshold=zero_threshold,
        merge_group=merge_group,
        source_roots_text=source_roots_text,
        merge_repo_id=merge_repo_id,
        merge_video_size_mb=merge_video_size_mb,
        ts_check_group=ts_check_group,
        video_key=video_key,
        tolerance_ms=tolerance_ms,
        ts_report_output=ts_report_output,
        ts_output_format=ts_output_format,
        v2_convert_group=v2_convert_group,
        v2_variant=v2_variant,
        v2_output_root=v2_output_root,
        v2_suffix=v2_suffix,
        cam_left_eye=cam_left_eye,
        cam_right_eye=cam_right_eye,
        cam_left_wrist=cam_left_wrist,
        cam_right_wrist=cam_right_wrist,
        v2_dry_run=v2_dry_run,
    )