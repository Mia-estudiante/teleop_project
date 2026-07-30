import asyncio
import websockets

clients = {}

async def handler(ws):
    path = ws.request.path
    print("connected:", path)

    clients[path] = ws
    try:
        async for msg in ws:
            if path == "/robot/head":
                viewer = clients.get("/viewer/head")
                if viewer:
                    await viewer.send(msg)
            elif path == "/viewer/head":
                robot = clients.get("/robot/head")
                if robot:
                    await robot.send(msg)
    finally:
        del clients[path]

async def run():
    async with websockets.serve(
        handler,
        "0.0.0.0",
        8765
    ):
        print("signaling server started")
        await asyncio.Future()

asyncio.run(run())