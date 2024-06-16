import asyncio
import base64
import cv2
import random
import tensorflow as tf
import websockets

from model import get_model


classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
roi = [150, 150, 400, 400]


def predict(model, image):
    image = image[150:400, 150:400, :]
    image = tf.image.resize(image, [48, 48])
    image = tf.image.rgb_to_grayscale(image)
    image = tr.cast(image, tf.float32) / 255.0
    batch = tf.expand_dims(image, axis=0)
    scores = model(batch)
    class_id = tf.math.argmax(scores)
    class_name = classes[class_id]
    return class_name


async def producer(queue: asyncio.Queue, frames_to_skip=20):
    model = get_model()
    cap = await asyncio.to_thread(cv2.VideoCapture, 0)
    if not cap.isOpened():
        raise RuntimeError("video capture is not open")
    else:
        print("video capture is open")
    while True:
        for _ in range(frames_to_skip):
            await asyncio.to_thread(cap.grab)
        ret, frame = await asyncio.to_thread(cap.read)
        # prediction = await asyncio.to_thread(predict, model, frame)
        # prediction = predict(model, frame)
        if not ret:
            raise RuntimeError("frame is not returned")
        _, image_buffer = cv2.imencode(".jpg", frame)
        encoded_image = base64.b64encode(image_buffer).decode("utf-8")
        await queue.put(encoded_image)
        # await queue.put({
        #     'image': encoded_image,
        #     'prediction': prediction
        # })
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