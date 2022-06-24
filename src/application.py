import tkinter as tk
import cv2 as cv
import PIL.Image, PIL.ImageTk
import time
import threading, queue

from GUI.toggle_button import ToggleButton
from GUI.scale_slider import ScaleSlider
from video_capture import VideoCapture
from ocr import OCR
from tkinter import ttk

class Application:

    ######################## INITIALIZATION ####################################################
    def __init__(self, window, window_title, args):
        self.window = window
        self.window.title(window_title)
        self.window.iconbitmap('../misc/cartooneye.ico')
        self.window.resizable(0, 0)
        self.window.configure(background='white')
        # Open video source (by default this will try to open the computer webcam).
        self.video_source = args.video_source
        self.vid = VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size.
        self.canvas = tk.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Style for custom widgets
        self.style = ttk.Style(window)
        self.style.configure('TLabel', font=('Lato', 12))
        self.style.configure('TButton', font=('Lato', 12))
        self.style.configure('TScale', font=('Lato', 7), background="white")

        # OCR
        self.language ='en'
        self.languages = {'Deutsch':'de', 'English':'en','Espagnol':'es', 'Français':'fr'}
        self.update_language = False
        self.ocr = OCR(args, self.vid.width, self.vid.height, self.language)
        self.ocr_task = None       # Will be a running thread performing the OCR.
        self.boxes = []            # Boxes drawn around words.
        self.pred_strings = []     # Predictions of words.
        self.spellchecks = []      # Suggestion of words. [(False, "suggestion"),..] if word is incorrect.
        self.det_infTime = 0.0     # Text detection model inference time.
        self.rec_infTime = 0.0     # Text recognition model inference time.
        self.pp_infTime = 0.0      # Postprocessing inference time.
        self.textRec = True        # Determines if we should run text recognition as well.
        self.blurThreshold = 100   # Used to determine if frame is sharp enough to perform OCR.
        self.blurry_frames_ctr = 0 # Used to count the number of concurrent blurry frames.
        self.new_conf_val = 0

        # For what to display
        self.display_boxes = False
        self.display_infTime = True
        self.display_predictions = False
        self.display_spellchecks = False

        # Create buttons etc.
        self.__create_duck_frame(self.window)
        self.__create_button_frame(self.window)
        self.__create_language_box(self.window)
        self.__create_sliders(self.window)

        # For snapshot
        self.save_frame = False

        # For multithreading
        self.q = queue.Queue()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()
    
    ####################################################################################################
    def __create_button_frame(self, container):
        """Creates the frame containing all the buttons."""
        frame = tk.Frame(container, background="white")

        # Buttons that lets the user toggle what to draw on the image.
        self.display_box_btn = ToggleButton(frame, self.toggle_displayBoxes, self.display_boxes)
        self.display_box_btn.grid(row=0, column=1)
        tk.Label(frame, text="Display Boxes", background="white").grid(row=0, column=0)

        self.display_inf_time_btn = ToggleButton(frame, self.toggle_displayInfTime, self.display_infTime)
        self.display_inf_time_btn.grid(row=1, column=1)
        tk.Label(frame, text="Display Inference Times", background="white").grid(row=1, column=0)

        self.text_rec_btn = ToggleButton(frame, self.toggle_textRec, self.textRec)
        self.text_rec_btn.grid(row=2, column=1)
        tk.Label(frame, text="Text Recognition", background="white").grid(row=2, column=0)

        self.display_pred_btn = ToggleButton(frame, self.toggle_displayPredictions, self.display_predictions)
        self.display_pred_btn.grid(row=3, column=1)
        tk.Label(frame, text="Display Predictions", background="white").grid(row=3, column=0)

        self.display_sc_btn = ToggleButton(frame, self.toggle_displaySpellchecks, self.display_spellchecks)
        self.display_sc_btn.grid(row=4, column=1)
        tk.Label(frame, text="Display Spellchecks", background="white").grid(row=4, column=0)

        # Button for taking a snapshot
        self.snap = tk.PhotoImage(file='../misc/snap.png')
        self.snap_btn = tk.Button(frame, image=self.snap, bd=0, background="white", command=self.snapshot)
        self.snap_btn.grid(row=5, column=1)

        # Put frame on right side of the container.
        frame.pack(side=tk.RIGHT, padx=20, expand=True)

    ####################################################################################################  
    def __create_duck_frame(self, container):
        global duck_photo
        frame = tk.Frame(container, background="white")

        duck_photo = tk.PhotoImage(file = "../misc/noduck.png")
        duck_btn = tk.Button(frame, image=duck_photo, bd=0, background="white", command=())
        duck_btn.pack(side=tk.TOP, pady=0, padx=5)

        # Put frame on left side of the container.
        frame.pack(side=tk.LEFT, padx=15, expand=True)

    ####################################################################################################
    def __create_language_box(self, container):
        """Creates the box where we pick the language."""
        # Add combobox for language selection

        language_combo = ttk.Combobox(container,
                                      values=[
                                          'Deutsch',
                                          'English',
                                          'Espagnol',
                                          'Français'],
                                      state="readonly")
        language_combo.pack(padx=1, pady=5, side=tk.BOTTOM)
        #language_combo.grid(row=4, column=4)
        language_combo.current(1)
        language_combo.bind("<<ComboboxSelected>>", self.switch_language)

        language_label = tk.Label(container, text='Language', background="white")
        language_label.pack(padx=1, pady=2, side=tk.BOTTOM)

    ####################################################################################################
    def __create_sliders(self, container):
        """Creates the GUI sliders."""
        # Style for custom slider widget
        global img_trough
        global img_slider
        img_trough = tk.PhotoImage(file = "../misc/trough.png")
        img_slider = tk.PhotoImage(file = "../misc/slider.png")
        self.style.element_create('custom.Scale.trough', 'image', img_trough)
        self.style.element_create('custom.Scale.slider', 'image', img_slider)
        self.style.layout('custom.Horizontal.TScale',
                     [('custom.Scale.trough', {'sticky': 'ew'}),
                      ('custom.Scale.slider',
                       {'side': 'left', 'sticky': '',
                        'children': [('custom.Horizontal.Scale.label', {'sticky': ''})]
                        })])
        # Add slider for blur threshold
        blur_label = tk.Label(container, text='Blur Threshold', background="white")
        blur_label.pack(padx=1, pady=1, side=tk.TOP)
        blur_scale = ScaleSlider(container, self.style, from_=0, to=500, command=self.set_blur_threshold)
        blur_scale.pack(padx=1, pady=1, side=tk.TOP)
        blur_scale.set(self.blurThreshold)

        # Add slider for min confidence threshold
        conf_label = tk.Label(container, text='Conf Threshold', background="white")
        conf_label.pack(padx=1, pady=1, side=tk.TOP)
        conf_scale = ScaleSlider(container, self.style, from_=0, to=10, command=self.set_conf_threshold)
        conf_scale.pack(padx=1, pady=1, side=tk.TOP)
        conf_scale.set(int(self.ocr.detector.conf * 10))

    ####################################################################################################
    def toggle_displayBoxes(self):
        self.display_boxes = not self.display_boxes
    
    def toggle_displayPredictions(self):
        self.display_predictions = not self.display_predictions

    def toggle_displaySpellchecks(self):
        self.display_spellchecks = not self.display_spellchecks

    def toggle_displayInfTime(self):
        self.display_infTime = not self.display_infTime

    def toggle_textRec(self):
        self.textRec = not self.textRec

    def switch_language(self, event):
        language = event.widget.get()
        self.language = self.languages[language]
        self.update_language = True

    def set_blur_threshold(self, value):
        self.blurThreshold = int(float(value))

    def set_conf_threshold(self, value):
        self.new_conf_val = int(float(value)) / 10.0 # conf should be between 0 and 1.

    ####################################################################################################
    def draw_boxes(self, frame):
        """Draws the bounding boxes around each detected word."""
        for box in self.boxes:
            frame = cv.rectangle(frame, (box[0], box[3]), (box[1], box[2]), (0, 255, 0), 1)

        return frame

    ###################################################################################################
    def draw_predicted_strings(self, frame):
        """Draws the recognized strings over each detected word."""
        for box, pred_string in zip(self.boxes, self.pred_strings):
            edge_dist = int((box[1]-box[0])*0.15)
            cv.putText(frame, pred_string, (box[0]+edge_dist, box[2]), cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

        return frame

    ####################################################################################################
    def draw_spellchecks(self, frame):
        """Draws found incorrect spelling and spell suggestions on the given frame."""
        for box, (word_incorrect, suggestion) in zip(self.boxes, self.spellchecks):
            xmin, xmax, ymin, ymax = box

            # If word is not correct we draw a red line and put a suggestion if it exists.
            if word_incorrect:
                edge_dist = int((xmax - xmin)*0.15)
                cv.line(frame, (xmin + edge_dist, ymax), (xmax - edge_dist, ymax), (100, 100, 250), 1)
                cv.putText(frame, suggestion, (xmin+edge_dist, ymax+10), cv.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 250))

        return frame

    ####################################################################################################
    def draw_inf_times(self, frame):
        """Draws the inference times on the given frame."""
        det_label = 'Detection: %.2f ms' % (self.det_infTime)
        rec_label = 'Recognition: %.2f ms' % (self.rec_infTime)
        pp_label = 'Postprocess: %.2f ms' % (self.pp_infTime)
        cv.putText(frame, det_label, (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        cv.putText(frame, rec_label, (0, 45), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        cv.putText(frame, pp_label, (0, 75), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))

        return frame

    ####################################################################################################
    def snapshot(self):
        self.save_frame = True

    ####################################################################################################
    def isBlurry(self, frame):
        # https://www.analyticsvidhya.com/blog/2020/09/how-to-perform-blur-detection-using-opencv-in-python/
        """Checks if frame is blurry. Returns true if blurry."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        fm = cv.Laplacian(gray, cv.CV_64F).var()

        return fm < self.blurThreshold

    ####################################################################################################
    def reset_OCR(self):
        self.boxes = []
        self.pred_corrections = []
        self.pred_strings = []
        self.det_infTime = 0
        self.rec_infTime = 0
        self.pp_infTime = 0

    ####################################################################################################
    def paint_OCR(self, frame):
        # If we want to draw the boxes around the words.
        if self.display_boxes:
            frame = self.draw_boxes(frame)

        # If we want to draw the predicted words.
        if self.display_predictions:
            frame = self.draw_predicted_strings(frame)
            
        # If we want to mark and suggest spelling corrections.
        if self.display_spellchecks:
            frame = self.draw_spellchecks(frame)

        # If we want to show inference times.
        if self.display_infTime:
            frame = self.draw_inf_times(frame)

        return frame
    
    ####################################################################################################
    def run_OCR(self, frame):
        """Starts a new thread that runs OCR on the given frame. First checks if some OCR related
        variables should be set before running."""
        # Check if we should update conf value.
        if self.new_conf_val != 0:
            self.ocr.set_minConf(self.new_conf_val)
            self.new_conf_val = 0 

        # Check if we should update language.
        if self.update_language:
            self.ocr.update_language(self.language)
            self.update_language = False

        # If the current frame is sharp, we can perform ocr on it.
        if not self.isBlurry(frame):
            self.blurry_frames_ctr = 0 # Reset counter of blurry frames.
            f = frame.copy()
            self.ocr_task = threading.Thread(target=self.ocr.perform_ocr, args=(f, self.q, self.textRec))
            self.ocr_task.start()
        else:
            self.blurry_frames_ctr += 1 # inc counter of blurry frames.

    ####################################################################################################
    def update(self):
        """The main loop of the application."""
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # If we have received multiple frames without clear text. Reset previous ocr predictions.
            if self.blurry_frames_ctr > 10:
                self.reset_OCR()

            # If we don't have an OCR already running, try to start a new one.
            if not self.ocr_task or not self.ocr_task.is_alive():
                self.run_OCR(frame)
                
            # Check if an OCR task has updated results.
            if not self.q.empty():
                self.det_infTime, self.rec_infTime, self.pp_infTime, self.boxes, self.pred_strings, self.spellchecks = self.q.get()
            
            frame = self.paint_OCR(frame)
            
            if self.save_frame:
                self.save_frame = False
                cv.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", frame)

            # Perform the necessary conversions so we can paint frame on self.canvas.
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)

        # Automatically call update method again after set delay.
        self.window.after(self.delay, self.update)
