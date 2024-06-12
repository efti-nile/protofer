import asyncio
import base64
import cv2
import random
import websockets


async def producer(queue: asyncio.Queue, frames_to_skip=20):
    cap = await asyncio.to_thread(cv2.VideoCapture, 0)
    if not cap.isOpened():
        raise RuntimeError("video capture is not open")
    else:
        print("video capture is open")
    while True:
        for _ in range(frames_to_skip):
            await asyncio.to_thread(cap.grab)
        ret, frame = await asyncio.to_thread(cap.read)
        if not ret:
            raise RuntimeError("frame is not returned")
        _, image_buffer = cv2.imencode(".jpg", frame)
        encoded_image = base64.b64encode(image_buffer).decode("utf-8")
        await queue.put(encoded_image)
        print("frame is put")


async def handler(queue, websocket, path):
    while True:
        item = await queue.get()
        if item is None:
            break
        await websocket.send(item)
        print("Sent item through WebSocket")


async def main():
    queue = asyncio.Queue()
    producer_task = asyncio.create_task(producer(queue))
    server = await websockets.serve(lambda ws, path: handler(queue, ws, path), "localhost", 8765)
    print("WebSocket server started on port 8765")
    await asyncio.Future()
    

# Run the main function
asyncio.run(main())