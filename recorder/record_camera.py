import datetime as dt
import sys
import os

import av
import cv2
import numpy as np
# import skvideo.io

MAX_PARTS = 2
PART_DURATION = dt.timedelta(seconds=30)
FOURCC = cv2.VideoWriter_fourcc(*'h264')
FPS = 20


class VideoWriter():

    def __init__(self, dstfile, fps, options={}):
        fps = int(round(fps))
        self.container = av.open(dstfile, mode="w")
        self.stream = self.container.add_stream("h264", rate=fps)
        self.stream.options = options
        self.init = False

    def write(self, frame):
        W, H = frame.shape[:2][::-1]
        if not self.init:
            self.stream.width = W
            self.stream.height = H
            self.init = True

        if self.stream.width != W or self.stream.height != H:
            raise Exception(f"Invalid frame shape: {(W,H)}, required: "
                            f"{(self.stream.width, self.stream.height)}")

        frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        packet = self.stream.encode(frame)
        self.container.mux(packet)

    def close(self):
        self.container.close()


def record_av(uri, label, out):
    cap = cv2.VideoCapture(uri)
    writer = None
    begin = None
    parts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            if writer:
                writer.close()
            raise Exception(f'no frame returned: {uri}')
        if not writer:
            h, w, _ = frame.shape
            begin = dt.datetime.now()
            video = f'{label}_{begin.strftime("%Y-%m-%dT%H-%M-%S")}.mp4'
            video_path = os.path.join(out, video)
            writer = VideoWriter(video_path, FPS, options={'codec': 'h264',
                                                           'bitrate': '3000k'})
        writer.write(frame)
        if dt.datetime.now() - begin > PART_DURATION:
            if writer:
                writer.close()
                writer = None
                parts += 1
        if parts > MAX_PARTS:
            cap.release()
            quit()


def play_av(uri):
    video = av.open(uri, 'r')
    try:
        for packet in video.demux():
            for frame in packet.decode():
                frame_array = np.array(frame.planes[0])
                cv2.imshow("video", frame_array)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


def record_ocv(uri, label, out):
    cap = cv2.VideoCapture(uri)
    writer = None
    begin = None
    parts = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            if writer:
                writer.release()
            raise Exception(f'no frame returned: {uri}')
        if not writer:
            h, w, _ = frame.shape
            begin = dt.datetime.now()
            video = f'{label}_{begin.strftime("%Y-%m-%dT%H-%M-%S")}.mp4'
            video_path = os.path.join(out, video)
            writer = cv2.VideoWriter(video_path, FOURCC, FPS, (w, h))
        writer.write(frame)
        if dt.datetime.now() - begin > PART_DURATION:
            if writer:
                writer.release()
                writer = None
                parts += 1
        if parts > MAX_PARTS:
            cap.release()
            quit()


if __name__ == '__main__':
    record_av(sys.argv[1], 'cam', 'data')
