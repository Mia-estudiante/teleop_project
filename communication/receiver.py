import asyncio
import json
import cv2
import websockets

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer
)

from aiortc.sdp import candidate_from_sdp

SIGNALING_URL = "ws://192.168.123.16:8765/viewer/head"

async def run():

    pc = RTCPeerConnection(
        RTCConfiguration([
            RTCIceServer(
                urls="stun:stun.l.google.com:19302"
            )
        ])
    )

    @pc.on("track")
    async def on_track(track):
        print("Receiving video")
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            cv2.imshow("WebRTC Viewer", img)
            if cv2.waitKey(1) == ord("q"):
                break

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state:", pc.connectionState)

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

        async for message in ws:
            data = json.loads(message)
            if data["type"] == "offer":
                print("Offer received")
                await pc.setRemoteDescription(
                    RTCSessionDescription(
                        sdp=data["sdp"],
                        type=data["type"]
                    )
                )

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await ws.send(json.dumps({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                }))
                print("Answer sent")

            elif data["type"] == "candidate":
                candidate = candidate_from_sdp(
                    data["candidate"]
                )
                candidate.sdpMid = data["sdpMid"]
                candidate.sdpMLineIndex = data["sdpMLineIndex"]
                await pc.addIceCandidate(candidate)

asyncio.run(run())