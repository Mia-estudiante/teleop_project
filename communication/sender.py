# robot_publisher.py
import asyncio
import json
import cv2
import numpy as np
import websockets

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    VideoStreamTrack
)

from aiortc.sdp import candidate_from_sdp
from av import VideoFrame

SIGNALING_URL = "ws://192.168.123.16:8765/robot/head"

class CameraStreamTrack(VideoStreamTrack):
    """RGB | mask side-by-side 송출"""
    def __init__(self, cam=0):
        super().__init__()
        self.cap = cv2.VideoCapture(cam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    async def recv(self):
        pts, tb = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            return None
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = tb

        # mask_vis = np.zeros_like(frame)
        # if target["name"]:
        #     for r in model.predict(frame, verbose=False, conf=0.35):
        #         if r.masks is None: continue
        #         for cls, m in zip(r.boxes.cls.cpu().numpy(),
        #                           r.masks.data.cpu().numpy()):
        #             if r.names[int(cls)].lower() == target["name"].lower():
        #                 m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
        #                 mask_vis[m > 0.5] = (0, 255, 0)

        # composite = np.hstack([frame, mask_vis])     # 1920x540
        # vf = VideoFrame.from_ndarray(composite, format="bgr24")
        # vf.pts, vf.time_base = pts, tb
        return new_frame

async def run():
    # stun 서버
    pc = RTCPeerConnection(
        RTCConfiguration([
            RTCIceServer(
                urls="stun:stun.l.google.com:19302"
            )
        ])
    )
    pc.addTrack(CameraStreamTrack())

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "connected":
            print("WebRTC connection established successfully")

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"Data channel established: {channel.label}")

    async with websockets.connect(SIGNALING_URL) as ws:
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is not None:
                await ws.send(json.dumps({
                    "type": "candidate",
                    "candidate": candidate.to_sdp(),
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex
                }))

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await ws.send(json.dumps({"sdp": pc.localDescription.sdp,
                                  "type": pc.localDescription.type}))
        # print(f"sdp: {pc.localDescription.sdp}, type: {pc.localDescription.type}")    
        print("Offer sent")

        async for message in ws:
            data = json.loads(message)
            if data["type"] == "answer":
                await pc.setRemoteDescription(
                    RTCSessionDescription(
                        sdp=data["sdp"],
                        type=data["type"]
                    )
                )
                print("Answer received")
            elif data["type"] == "candidate":
                candidate = candidate_from_sdp(
                    data["candidate"]
                )
                candidate.sdpMid = data["sdpMid"]
                candidate.sdpMLineIndex = data["sdpMLineIndex"]
                await pc.addIceCandidate(candidate)
                print("ICE candidate added")

        # ans = json.loads(await ws.recv())
        # await pc.setRemoteDescription(RTCSessionDescription(**ans))

asyncio.run(run())