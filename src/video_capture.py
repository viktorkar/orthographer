import cv2 as cv

class VideoCapture:

    def __init__(self, source=0):
        self.cap = cv.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", source)

        # Get video source width and height
        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    #####################################################################
    def get_frame(self):
        """Get a single frame from the video source."""
        if self.cap.isOpened():
            return self.cap.read()
        else:
            return (False, None)

    #####################################################################
    def __del__(self):
        """Release the video source when the object is destroyed."""
        if self.cap.isOpened():
            self.cap.release()