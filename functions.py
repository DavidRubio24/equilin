import logging
import os.path
import sys
import time

import cv2
import numpy as np
from scipy import fftpack, signal

from utils.worker import KeepThemComing


log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stderr))
log.handlers[-1].setFormatter(logging.Formatter('\x1B[0;34m%(asctime)s - %(name)s.%(funcName)s\t- %(levelname)s - %(message)s\x1B[0m'))
log.setLevel(logging.DEBUG)


class VideoStreamReader:
    def __init__(self, video_input, duration=None):
        self.capture = cv2.VideoCapture(video_input)
        self.time_duration   = duration if isinstance(duration, float) else float('inf')
        self.frames_duration = duration if isinstance(duration,   int) else float('inf')
        self.until = None
        self.frame_count = 0
        self.__dict__['names'] = ('id', 'image')

    def __iter__(self): return self

    def __next__(self):
        success, image = self.capture.read()
        timestamp = time.monotonic()
        self.until = self.until or timestamp + self.time_duration
        if self.frame_count >= self.frames_duration or timestamp > self.until or not success:
            self.capture.release()
            raise StopIteration
        self.frame_count += 1
        return timestamp, image[..., ::-1]

    __call__ = __next__


class ImageDirectoryReader:
    def __init__(self, path, delay=1/140):
        if not os.path.isdir(path):
            raise ValueError(f'{path} is not a directory.')
        self.images = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
        if not self.images:
            raise ValueError(f'{path} has no PNGs.')
        self.delay = delay
        self.next_image = 0
        self.last = time.monotonic()
        self.names = ('id', 'image')

    def __iter__(self): return self

    def __next__(self):
        if self.next_image >= len(self.images):
            raise StopIteration()

        # Wait for the delay since last time.
        now  = time.monotonic()
        if now - self.last < self.delay:
            time.sleep(self.delay - (now - self.last))
        self.last = time.monotonic()

        image = cv2.imread(self.images[self.next_image])
        self.next_image += 1
        if image is None:
            return self.__next__()
        return self.next_image, image[..., ::-1]

    __call__ = __next__

    def __len__(self): return len(self.images)


def get_results_from_detector(detector, image) -> list:
    """Unified way of getting results from different types of detectors (mainly FaceMesh and Hands)."""
    processed = detector.process(image)
    return [processed.__dict__[field] for field in processed._fields]


def landmarks(image, detector):
    if image is None:
        raise ValueError('Image {id} is None.')

    results = get_results_from_detector(detector, image)

    if results[0] is None:
        raise ValueError(f'No detected {detector.__class__.__name__} in {{id}}.')

    # MediaPipe returns landmarks in an obscure format. Convert them to np.ndarray.
    landmarks = np.array([(l.x, l.y, l.z) for l in results[0][0].landmark])
    landmarks[..., :2] *= (image.shape[1], image.shape[0])  # From [0, 1] to image size.

    return landmarks


def contour(landmarks, points_indexes, combination=None) -> np.ndarray:
    """
    Return contour of the ROI based on the landmarks.

    :param landmarks: landmarks of the face or hand.
    :param points_indexes: indexes of the points to use.
    :param combination: combination of the points to use. If None, use all points.
                        If it's a matrix, use the linear combination given by it.
    :return: array of pixel positions of the points detimiting the contour of the ROI.
    """
    contour = landmarks[list(points_indexes)]
    if combination is not None:
        contour = np.einsum('mn,nk->mk', np.array(combination), contour)
    return contour


def roi(image, landmarks, contour=None, points_indexes=None, excludes=(), blur=(5, 5)):
    # Get mask from contour.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if contour is None:
        if points_indexes is not None:
            contour = landmarks[list(points_indexes)]
        else:
            raise ValueError('Either contour or points_indexes must be given.')
    cv2.drawContours(mask, [np.round(contour[..., :2]).astype(int)], 0, 255, -1)

    # Remove excluded contours.
    for exclude in excludes:
        exclude = landmarks[list(exclude)][..., :2]
        cv2.drawContours(mask, [np.round(exclude).astype(int)], 0, 0, -1)

    # TODO: Remove pixels that are obviusly not skin.

    if blur:
        # Blur the mask.
        mask = cv2.GaussianBlur(mask, blur, 0)
    return mask


def roi_int(image, landmarks, contour=None, points_indexes=None, excludes=(), blur=(5, 5)):
    # Get mask from contour.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if contour is None:
        if points_indexes is not None:
            contour = landmarks[list(points_indexes)]
        else:
            raise ValueError('Either contour or points_indexes must be given.')
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # Remove excluded contours.
    for exclude in excludes:
        exclude = landmarks[list(exclude)]
        cv2.drawContours(mask, [exclude], 0, 0, -1)

    # TODO: Remove pixels that are obviusly not skin.

    if blur:
        # Blur the mask.
        mask = cv2.GaussianBlur(mask, blur, 0)
    return mask


# def ppg(image, roi, weighted=True):
#     weights = np.repeat(roi[..., np.newaxis], repeats=image.shape[-1], axis=-1)
#     if weighted:
#         mean = np.average(image, weights=weights, axis=(0, 1))
#     else:
#         mean = np.mean(image[roi.astype(bool)], axis=0)
#     return mean


def ppg(image, roi, min_x=0, min_y=0, max_x=-1, max_y=-1):
    return np.mean(image[min_y:max_y, min_x:max_x][roi[min_y:max_y, min_x:max_x].astype(bool)], axis=0)


def gaussian(h, w, sigma=10) -> np.ndarray:
    # TODO: compute as separate 1D filters.
    kernel = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            kernel[i, j] = np.exp(-((i - h // 2) ** 2 + (j - w // 2) ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


KERNEL = gaussian(40, 40, 15)


def ppg_around(image, landmarks, kernel=KERNEL):
    values = []
    kernel_ = kernel
    h, w = kernel.shape[:2]
    h2, w2 = h // 2, w // 2

    for x, y in np.round(landmarks[..., :2]).astype(int):
        x0, y0, x1, y1 = x - h2, y - w2, x + h2, y + w2
        kernel = kernel_

        # Make sure the kernel is inside the image.
        if y0 < 0: kernel = kernel[-y0:, :]
        if x0 < 0: kernel = kernel[:, -x0:]
        if y1 > image.shape[0]: kernel = kernel[:image.shape[0] - y1, :]
        if x1 > image.shape[1]: kernel = kernel[:, :image.shape[1] - x1]

        for channel in np.split(image, image.shape[-1], axis=-1):
            value = kernel * channel[y0:y1, x0:x1].squeeze()
            values.append(np.sum(value))
    return np.array(values).flatten()


def draw(image, landmarks, color=(255, 255, 255), radius=2, copy=True):
    image = image.copy() if copy else image
    for x, y in np.round(landmarks[..., :2]).astype(int):
        cv2.circle(image, (x, y), radius, color, cv2.FILLED)
    return image


def draw_contour(image, contour, color=(255, 255, 255), thickness=2, copy=True):
    image = image.copy() if copy else image
    cv2.drawContours(image, [np.round(contour[:, :2]).astype(int)], 0, color, thickness)
    return image


KERNEL = gaussian(40, 40, 2)
KERNEL /= KERNEL.max()
KERNEL = np.repeat(KERNEL[..., np.newaxis], repeats=3, axis=-1)


def draw_kernel(image, landmarks, copy=True, kernel=KERNEL):
    image = image.copy() if copy else image
    kernel_ = kernel
    h, w = kernel.shape[:2]
    h2, w2 = h // 2, w // 2

    for x, y in np.round(landmarks[..., :2]).astype(int):
        x0, y0, x1, y1 = x - h2, y - w2, x + h2, y + w2
        kernel = kernel_

        # Make sure the kernel is inside the image.
        if y0 < 0: kernel = kernel[-y0:, :]
        if x0 < 0: kernel = kernel[:, -x0:]
        if y1 > image.shape[0]: kernel = kernel[:image.shape[0] - y1, :]
        if x1 > image.shape[1]: kernel = kernel[:, :image.shape[1] - x1]

        # Reduce the ditance between the pixels and white acording to the kernel.
        image[y0:y1, x0:x1] = 255 - (255 - image[y0:y1, x0:x1]) * (1 - kernel)

    return image


class Save:
    def __init__(self, path):
        self.path = path
        self.values = []

    def __call__(self, ppg):
        self.values.append(ppg)

    def __del__(self):
        np.save(self.path, self.values)


class SaveCSV(Save):
    def __del__(self):
        np.savetxt(self.path, self.values, delimiter=',')


class Interpolate:
    def __init__(self):
        self.last_landmarks = None
        self.last_id = None
        self.ids = []

    def __call__(self, id, landmarks=None):
        # We'll have to interpolate this one later.
        if landmarks is None:
            self.ids.append(id)
            raise KeepThemComing(True)
        # We have new landmarks, but there are no ids to intepolate.
        elif not self.ids:
            # Save them as the most up-to-date ones.
            self.last_landmarks = landmarks
            self.last_id = id
            return landmarks
        # We have new landmarks and previous ids to intepolate.
        else:
            span = id - self.last_id
            # For each stored id, interpolate the landmarks.
            interpolateds = []
            for id_ in self.ids:
                alpha = (id_ - self.last_id) / span
                interpolated = (1 - alpha) * self.last_landmarks   +   alpha * landmarks
                interpolateds.append(interpolated)
            interpolateds.append(landmarks)
            self.last_landmarks = landmarks
            self.last_id = id
            self.ids.clear()
            return interpolateds


def show(image, title='Image', delay=1, roi=None):
    if len(image.shape) == 2:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)

    if roi is not None:
        # Black out the pixels outside the ROI.
        image[np.logical_not(roi.astype(bool))] = [0, 0, 0]

    # Show image mirrored and converted to BGR.
    cv2.imshow(title, image[..., ::-1, ::-1])
    cv2.waitKey(delay)


def print_ppg(ppg):
    print(ppg)


class SaveVideo:
    def __init__(self, path, shape=(480, 640), fps=30, fourcc='HFYU'):
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, shape[:2][::-1])

    def __call__(self, image): self.writer.write(image[..., ::-1])

    def __del__(self): self.writer.release()


def get_pulse(ppg: np.ndarray, lowest_pulse=50, highest_pulse=150, fps=None, secs=80, best_of=6) -> float:
    """
    Get the pulse from the PPG.

    :param ppg: The PPG signal. It would be better if its length is a power of 2.
    :param lowest_pulse: The lowest pulse that we'll accept, mesured in beats per minute.
    :param highest_pulse: The highest pulse that we'll accept, mesured in beats per minute.
    :param fps: The frequency of the PPG, in frames per seconds. By default (None), it's calculated from the seconds.
    :param secs: The number of seconds that last this PPG signal. It's ignored if fpm is not None.
    :param best_of: The number of best frequencies to average.

    :return: The pulse, in beats per minute.
    """
    # Get the Fourier Transform of the PPG.
    fft = fftpack.fft(ppg, axis=0)
    """
    The FFT of a real signal is a complex array, where:
    fft[i] is the conjugate of fft[-i]. Hence, fft[0] = fft[-0] is real.
    fft[i] represents the i-th frequency, i.e. the frequency that would have i peaks in the signal (given its length).
    abs(fft[i]) is the amplitude of the i-th frequency multiplied by half of its length. The other half is in fft[-i].
    fft[0] is the sum of all the signal's values, beacuse the 0-th frecuency is the mean.
    fft[i]'s angle is the phase of the i-th frequency. A cosine is angle 0     (positive real),
                                                       a sinus  is angle -pi/2 (negative imaginary).
    """

    # Compute the frames per minute from the frames per second or the seconds.
    fpm = 60 * (fps or round(len(ppg) / secs))

    # Get the frequency of the lowest and highest pulse we'll admit.
    # If there are lowest_pulse beats per minute and fpm samples per minute,
    # then there's a pulse every fpm / lowest_pulse samples.
    lowest_pulse_freq  = len(ppg) * lowest_pulse  // fpm
    highest_pulse_freq = len(ppg) * highest_pulse // fpm

    # We are only interested in the amplitude of certain range of frequencies.
    freqs_of_interest = abs(fft[lowest_pulse_freq:highest_pulse_freq])
    # Take the indexes of the bigest ones.
    biggest_frequencies = np.argpartition(-freqs_of_interest, min(best_of, len(freqs_of_interest)) - 1)[:best_of]
    log.debug((biggest_frequencies + lowest_pulse_freq) * fpm / len(ppg))

    # Those indexes are the frequencies of the best pulses. Take the weighted mean.
    # To weight them, take into account their amplitude.
    amplitude = freqs_of_interest[biggest_frequencies]
    # Also, take into account the distance from their average.
    distances_to_avg = abs(biggest_frequencies - np.average(biggest_frequencies, weights=amplitude)) + .1
    pulse = np.average(biggest_frequencies, weights=amplitude ** 2 / distances_to_avg)
    # But the indexes are frequencies relative to the whole fft, not respect the range we are interested in.
    pulse += lowest_pulse_freq
    # Convert the frequency to beats per minute.
    pulse *= fpm / len(ppg)

    return pulse


def amplify_frequencies(fft, amplify, lowest_pulse=50, highest_pulse=150, fpm=1800, copy=False):
    """
    Amplify the frequencies of the PPG.

    :param fft: The Fourier Transform of the PPG.
    :param amplify: The factor by which to amplify the frequencies.
    :param lowest_pulse: The lowest pulse that we'll accept, mesured in beats per minute.
    :param highest_pulse: The highest pulse that we'll accept, mesured in beats per minute.
    :param fpm: The frequency of the PPG, in frames per minute.
    :param copy: Whether to copy the fft or not.

    :return: The Fourier Transform of the PPG.
    """
    fft = fft.copy() if copy else fft

    # Get the frequency of the lowest pulse.
    # There are lowest_pulse beats per minute and fpm samples per minute,
    # i.e. a pulse every fpm / lowest_pulse samples.
    lowest_pulse_freq  = len(fft) * lowest_pulse  // fpm
    # Get the frequency of the highest pulse.
    highest_pulse_freq = len(fft) * highest_pulse // fpm

    # We are only interested in amplifying certain ranges of frequencies.
    fft[lowest_pulse_freq:highest_pulse_freq]   *= amplify
    fft[len(fft) - highest_pulse_freq + 1: len(fft) - lowest_pulse_freq + 1] *= amplify

    return fft


def isolate_frequencies(fft, lowest_pulse=50, highest_pulse=150, fps=None, secs=80, copy=False):
    """
    Isolate the frequencies of the PPG.

    :param fft: The Fourier Transform of the PPG.
    :param lowest_pulse: The lowest pulse that we'll accept, mesured in beats per minute.
    :param highest_pulse: The highest pulse that we'll accept, mesured in beats per minute.
    :param fps: The frequency of the PPG, in frames per second.
    :param secs: The number of seconds that last this PPG signal. It's ignored if fpm is not None.
    :param copy: Whether to copy the fft or not.

    :return: The Fourier Transform of the PPG.
    """
    fft = fft.copy() if copy else fft
    fps = fps or round(len(fft) / secs)

    # Get the frequency of the lowest and highest pulse.
    # If there are lowest_pulse beats per minute and fpm samples per minute,
    # i.e. a pulse every fpm / lowest_pulse samples.
    lowest_pulse_freq  = len(fft) * lowest_pulse // (60 * fps)
    highest_pulse_freq = len(fft) * highest_pulse // (60 * fps)

    # Zero the frequencies we're not interested in.
    fft[:lowest_pulse_freq] = 0
    fft[len(fft) - lowest_pulse_freq + 1:] = 0
    fft[highest_pulse_freq: len(fft) - highest_pulse_freq + 1] = 0

    return fft


def get_peaks(ppg, lowest_pulse=50, highest_pulse=150, fps=None, secs=80):
    """
    Get the peaks of the PPG.

    :param ppg: The PPG signal.
    :param lowest_pulse: The lowest pulse that we'll accept, mesured in beats per minute.
    :param highest_pulse: The highest pulse that we'll accept, mesured in beats per minute.
    :param fps: The frequency of the PPG, in frames per second.
    :param secs: The number of seconds that last this PPG signal. It's ignored if fpm is not None.

    :return: The peaks of the PPG.
    """
    # Get the Fourier Transform of the PPG.
    fft = fftpack.fft(ppg, axis=0)

    fps = fps or round(len(ppg) / secs)

    # Get the frequency of the lowest and highest pulse.
    # If there are lowest_pulse beats per minute and fpm samples per minute,
    # i.e. a pulse every fpm / lowest_pulse samples.
    lowest_pulse_freq  = len(ppg) * lowest_pulse  // (60 * fps)
    highest_pulse_freq = len(ppg) * highest_pulse // (60 * fps)

    # Zero the frequencies we're not interested in.
    fft[:lowest_pulse_freq] = 0
    fft[len(fft) - lowest_pulse_freq + 1:] = 0
    fft[highest_pulse_freq: len(fft) - highest_pulse_freq + 1] = 0

    ifft = fftpack.ifft(fft, axis=0)

    # Get the peaks of the PPG.
    peaks, _ = signal.find_peaks(ifft.real)

    return peaks
