# Camera Stream Class with Threading

from threading import Thread, Lock
import cv2


class CamStream:
    # Ctor
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.mutex = Lock()
        self.started = False
        self.thread = None

    # Dtor
    def __del__(self):
        self.stream.release()

    def start(self):
        if self.started:
            print('Already Started')
            return None
        else:
            self.started = True
            self.thread = Thread(target=self.update, args=())
            self.thread.start()
            return self

    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if grabbed:
                self.mutex.acquire()
                self.grabbed, self.frame = grabbed, frame
                self.mutex.release()
            cv2.waitKey(1)

    def read(self):
        self.mutex.acquire()
        grabbed, frame = self.grabbed, self.frame
        self.mutex.release()
        return grabbed, frame

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()


if __name__ == '__main__':
    cs = CamStream().start()
    winname = 'test_window'
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    while True:
        grab, frame = cs.read()
        cv2.imshow(winname, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cs.stop()
    cv2.destroyAllWindows()
