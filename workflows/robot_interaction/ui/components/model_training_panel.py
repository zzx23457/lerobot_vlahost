"""Model training panel factory for robot_interaction UI.

创建一个 Gradio Accordion，包含 ACT / SmolVLA / Fine-tune 训练的所有控件。
返回 dataclass，便于 app_zh.py 解包访问组件并接入事件。
"""

from dataclasses import dataclass
from pathlib import Path
import gradio as gr

from .common import create_file_picker


@dataclass
class ModelTrainingPanel:
    """打包所有模型训练相关组件。"""

    panel: gr.Accordion
    script: gr.Radio
    phase: gr.Radio
    dataset_root: gr.Dropdown  # 从 datasets/ 自动列出
    dataset_root_refresh_btn: gr.Button
    output_root: gr.Textbox  # 只读显示,使用脚本默认
    # optimization (通用)
    opt_group: gr.Group
    batch_size: gr.Number
    steps: gr.Number
    eval_freq: gr.Number
    save_freq: gr.Number
    log_freq: gr.Number
    # tracking
    trk_group: gr.Group
    wandb_project: gr.Textbox
    wandb_enable: gr.Checkbox
    push_to_hub: gr.Checkbox
    # smolvla
    smolvla_group: gr.Group
    policy_chunk_size: gr.Number
    policy_n_action_steps: gr.Number
    policy_lr: gr.Number
    policy_path: gr.Textbox
    load_vlm_weights: gr.Checkbox
    freeze_vision_encoder: gr.Checkbox
    train_expert_only: gr.Checkbox
    hf_endpoint: gr.Textbox
    rename_map_json: gr.Textbox
    # finetune
    finetune_group: gr.Group
    pretrained_ckpt: gr.Textbox
    new_dataset: gr.Textbox


def create_model_training_panel() -> ModelTrainingPanel:
    """Create the model training panel and return a dataclass of all widgets.

    Visibility groups (smolvla / finetune) are returned but their visibility
    is driven by ``app_zh.py``'s ``on_training_script_change``.
    """
    # 延迟导入避免循环依赖
    from workflows.robot_interaction.ui.app_zh import _list_datasets_full
    # 本地计算仓库根: components/ → ui/ → robot_interaction/ → workflows/ → <REPO_ROOT>
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
    with gr.Accordion("模型训练", open=True, visible=False) as panel:
        gr.Markdown(
            "**📦 训练通过 `bash train_*.sh <phase>` 启动**。"
            "所有环境变量走白名单注入；不会污染 Gradio 主进程。"
        )

        with gr.Row():
            script = gr.Radio(
                choices=[
                    ("ACT", "act"),
                    ("SmolVLA", "smolvla"),
                    ("ACT Fine-tune", "finetune"),
                ],
                value="act",
                label="训练脚本",
                scale=1,
            )
            phase = gr.Radio(
                choices=["all", "env", "check", "smoke", "train", "eval"],
                value="smoke",
                label="阶段",
                info="smoke=小步数验证 / train=正式训练 / eval=仅打印命令",
                scale=2,
            )

        with gr.Row():
            dataset_root = gr.Dropdown(
                label="DATASET_ROOT (训练数据集路径, 从 datasets/ 自动列出)",
                choices=_list_datasets_full(),
                value=None,
                allow_custom_value=True,
                info="下拉选择或在文本框里输入完整路径",
                scale=4,
            )
            dataset_root_refresh_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)
        # OUTPUT_ROOT 仅显示默认路径,不可编辑 (脚本端已为它准备默认值 $PROJECT_ROOT/outputs/train)
        output_root = gr.Textbox(
            value=str(_REPO_ROOT / "outputs" / "train"),
            label="OUTPUT_ROOT (checkpoint 目录, 由脚本默认决定, 不可编辑)",
            interactive=False,
            info="训练脚本内置默认 ${PROJECT_ROOT}/outputs/train,无需修改",
        )

        # Optimization (通用 ACT / SmolVLA)
        with gr.Group() as opt_group:
            gr.Markdown("**训练超参 (optimization)**")
            with gr.Row():
                batch_size = gr.Number(value=8, label="BATCH_SIZE", precision=0, minimum=1)
                steps = gr.Number(value=400000, label="STEPS", precision=0, minimum=1)
                log_freq = gr.Number(value=50, label="LOG_FREQ", precision=0, minimum=1)
            with gr.Row():
                eval_freq = gr.Number(value=20000, label="EVAL_FREQ", precision=0, minimum=1)
                save_freq = gr.Number(value=20000, label="SAVE_FREQ", precision=0, minimum=1)

        # Tracking (通用 ACT / SmolVLA)
        with gr.Group() as trk_group:
            gr.Markdown("**日志 & 上传 (tracking)**")
            wandb_project = gr.Textbox(
                label="WANDB_PROJECT",
                placeholder="留空使用脚本默认",
            )
            with gr.Row():
                wandb_enable = gr.Checkbox(value=False, label="WANDB_ENABLE")
                push_to_hub = gr.Checkbox(value=False, label="PUSH_TO_HUB")

        # SmolVLA 专属
        with gr.Group(visible=False) as smolvla_group:
            gr.Markdown("**SmolVLA 专属参数**")
            with gr.Row():
                policy_chunk_size = gr.Number(value=50, label="POLICY_CHUNK_SIZE", precision=0, minimum=1)
                policy_n_action_steps = gr.Number(value=50, label="POLICY_N_ACTION_STEPS", precision=0, minimum=1)
                policy_lr = gr.Number(value=1e-4, label="POLICY_LR", minimum=0)
            policy_path = gr.Textbox(
                label="POLICY_PATH (HF repo 或本地路径)",
                placeholder="lerobot/smolvla_base",
            )
            with gr.Row():
                load_vlm_weights = gr.Checkbox(value=False, label="LOAD_VLM_WEIGHTS")
                freeze_vision_encoder = gr.Checkbox(value=True, label="FREEZE_VISION_ENCODER")
                train_expert_only = gr.Checkbox(value=True, label="TRAIN_EXPERT_ONLY")
            hf_endpoint = gr.Textbox(
                label="HF_ENDPOINT (国内镜像)",
                placeholder="https://hf-mirror.com",
            )
            rename_map_json = gr.Textbox(
                label="RENAME_MAP (JSON 字符串, 紧凑无空格)",
                placeholder='{"right_eye":"camera1"}',
                lines=2,
            )

        # Fine-tune 专属
        with gr.Group(visible=False) as finetune_group:
            gr.Markdown("**ACT Fine-tune 专属参数**")
            pretrained_ckpt = create_file_picker(
                label="PRETRAINED_CKPT (已训练 checkpoint)",
                placeholder="outputs/train/act_*/checkpoints/200000/pretrained_model",
            )
            new_dataset = create_file_picker(
                label="NEW_DATASET (新数据集路径)",
                placeholder="datasets/new_data",
            )

    return ModelTrainingPanel(
        panel=panel,
        script=script,
        phase=phase,
        dataset_root=dataset_root,
        dataset_root_refresh_btn=dataset_root_refresh_btn,
        output_root=output_root,
        opt_group=opt_group,
        batch_size=batch_size,
        steps=steps,
        eval_freq=eval_freq,
        save_freq=save_freq,
        log_freq=log_freq,
        trk_group=trk_group,
        wandb_project=wandb_project,
        wandb_enable=wandb_enable,
        push_to_hub=push_to_hub,
        smolvla_group=smolvla_group,
        policy_chunk_size=policy_chunk_size,
        policy_n_action_steps=policy_n_action_steps,
        policy_lr=policy_lr,
        policy_path=policy_path,
        load_vlm_weights=load_vlm_weights,
        freeze_vision_encoder=freeze_vision_encoder,
        train_expert_only=train_expert_only,
        hf_endpoint=hf_endpoint,
        rename_map_json=rename_map_json,
        finetune_group=finetune_group,
        pretrained_ckpt=pretrained_ckpt,
        new_dataset=new_dataset,
    )