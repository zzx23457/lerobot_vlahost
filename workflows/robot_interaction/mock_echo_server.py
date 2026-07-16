#!/usr/bin/env python3
"""mock_echo_server.py — 假机器人 HTTP server,用于观察客户端真实上线的 action / chunk。

它不连任何真机,只做两件事:
  1. GET /state          返回合成观测(14 关节 + 双夹爪 + base64 quad 图),
                         让 rollout 循环能正常产生动作。
  2. POST /action        打印单帧 body,返回 {"success": true}
     POST /action_chunk  打印整块 body(条数 + JSON),返回 {"success": true}

用法:
    # 1) 起 mock(默认 0.0.0.0:8010)
    python workflows/robot_interaction/mock_echo_server.py --port 8010

    # 2) 另开一个终端跑部署,把 http_base_url 指到本机 mock
    python workflows/robot_interaction/deploy.py \
        --http-base-url http://127.0.0.1:8010 \
        --fps 2            # 降频,print 才看得清

依赖:fastapi / uvicorn / opencv-python / numpy(仓库里都有)。
"""
import argparse
import base64
import json

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="mock echo server")

# ---- 预生成一张合成 quad 图(1280x960 = 2x2 的 640x480),避免每次 GET 都重编码 ----
# 画成 4 个不同灰度块,保证 right_eye 不是全黑(否则客户端会 fallback 成 left_eye)。
_quad = np.zeros((960, 1280, 3), dtype=np.uint8)
_quad[0:480, 0:640] = 40      # tl
_quad[0:480, 640:1280] = 80   # tr
_quad[480:960, 0:640] = 120   # bl
_quad[480:960, 640:1280] = 160  # br
_ok, _enc = cv2.imencode(".jpg", _quad, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
_QUAD_B64 = base64.b64encode(_enc.tobytes()).decode("ascii")


@app.get("/state")
def get_state():
    """返回一份能让 rollout 正常取观测的合成 /state。"""
    return {
        "stamp": 0,
        "joint_states": {
            "positions": [0.0] * 14,   # 14 个手臂关节(弧度)
            "velocities": [0.0] * 14,
            "efforts": [0.0] * 14,
        },
        "gripper_left": {"position": 0.0},   # dict 格式,客户端取 ["position"]
        "gripper_right": {"position": 0.0},
        # 用旧 base64 格式(客户端有 "data" 分支),省得再实现 MJPEG 流
        "quad_image": {"format": "jpeg", "data": _QUAD_B64},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/action")
async def post_action(payload: dict):
    """单帧:打印 body。"""
    print("\n===== POST /action =====")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return {"success": True}


@app.post("/action_chunk")
async def post_action_chunk(payload: dict):
    """整块:打印条数 + 完整 JSON。"""
    actions = payload.get("actions", [])
    print("\n===== POST /action_chunk =====")
    print(f"source_hz = {payload.get('source_hz')}   |   n_actions = {len(actions)}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return {"success": True, "n_actions": len(actions)}


def main() -> None:
    parser = argparse.ArgumentParser(description="mock echo server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    print(f"mock echo server on http://{args.host}:{args.port}  (GET /state, POST /action, POST /action_chunk)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
