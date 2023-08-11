import datetime as dt
import sys
import os

import av
import cv2
import numpy as np
# import skvideo.io

MAX_PARTS = 100
PART_DURATION = dt.timedelta(minutes=5)
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 20


def record_av(uri, label):
    fh = av.open(uri)
    print(fh.format)
    print(fh.streams)
    print(fh.streams[0].format)


# def play_skv(uri):
#     videodata = skvideo.io.vread("video_file_name")  
#     print(videodata.shape)


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
        if dt.datetime.now() - begin > PART_DURATION:
            if writer:
                writer.release()
                writer = None
                parts += 1
        if parts > MAX_PARTS:
            cap.release()
            quit()


if __name__ == '__main__':
    record_ocv(sys.argv[1], 'cam', 'data')
