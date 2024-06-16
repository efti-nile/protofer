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
    try:
        image = image[150:400, 150:400, :]
        image = tf.image.resize(image, [48, 48])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32) / 255.0
        batch = tf.expand_dims(image, axis=0)
        scores = model(batch)
        class_id = tf.math.argmax(scores)
        class_name = classes[class_id]
        return class_name
    except Exception as e:
        print(f"Exception in predict: {e}")
        raise


async def producer(queue: asyncio.Queue, frames_to_skip=20):
    try:
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
            await queue.put(frame.copy())
            print("frame is put")
    except Exception as e:
        print(f"Exception in producer: {e}")
        raise


async def processor(queue_in: asyncio.Queue,
                    queue_out: asyncio.Queue):
    try:
        model = get_model()
        while True:
            frame = await queue.get()
            prediction = await asyncio.to_thread(predict, model, frame)
            # TODO 1. add image encoding 2. fix glagnu
            await queue_out.put({
                'frame': frame.copy(),
                'prediction': prediction
            })
            print('frame processed')
    except Exception as e:
        print(f"Exception in processor: {e}")
        raise


async def handler(queue, websocket, path):
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            await websocket.send(item)
            print("Sent item through WebSocket")
    except Exception as e:
        print(f"Exception in handler: {e}")
        raise


async def main():
    queue = asyncio.Queue()
    producer_task = asyncio.create_task(producer(queue))
    processor_task = asyncio.create_task(processor(queue))
    await asyncio.gather(producer_task, processor_task)
    # Uncomment the following lines to run the WebSocket server
    # server = await websockets.serve(lambda ws, path: handler(queue, ws, path), "localhost", 8765)
    # print("WebSocket server started on port 8765")
    # await asyncio.Future()


# Run the main function
asyncio.run(main(), debug=True)
