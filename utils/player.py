# Run withouth arguments to play all videos in the current directory.
# Run with arguments to play only those videos.

import os
import sys

import cv2

VIDEO_FORMATS = ('.mjpeg', '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.mpg', '.mpeg', '.m4v', '.mjpg', '.webm')


def play(video, delay=10):
    cap = cv2.VideoCapture(video)
    success, image = cap.read()
    while success:
        cv2.imshow(video, image)
        key = cv2.waitKey(delay)
        if key == 27:  # ESC
            break
        success, image = cap.read()
    cv2.destroyWindow(video)
    cap.release()


def main(*videos):
    videos = videos or os.listdir()
    for video in videos:
        if video.endswith(VIDEO_FORMATS):
            play(video)
        else:
            print(f"{video} doesn't have a video extension.", end='')
            insist = input(" Play anyway? (y/[n]) ")
            if insist.lower() in ['y', 'yes', 'si', 's', 'sí', 'ok', 'okay']:
                play(video)


if __name__ == '__main__':
    main(*sys.argv[1:])
