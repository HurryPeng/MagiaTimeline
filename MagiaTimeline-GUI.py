import tkinter as tk
from tkinter import filedialog
import customtkinter
from PIL import Image, ImageTk, Image
import typing
import av
import av.container
import av.video
import fractions
import random
import sys
import multiprocessing
import threading
import json
import yaml
import tempfile

from Version import VERSION

# Initialize appearance
customtkinter.set_appearance_mode("Dark")  # Modes: System, Dark, Light
customtkinter.set_default_color_theme("blue")  # Themes: blue, green, dark-blue

class VideoPlayer:
    def __init__(self, path):
        # Open container and video stream
        self.path = path
        self.container: av.container.InputContainer = av.open(path, mode='r')
        self.stream: av.video.stream.VideoStream = self.container.streams.video[0]
        self.stream.thread_type = 'FRAME'
        # Video dimensions
        width = self.stream.codec_context.width
        height = self.stream.codec_context.height
        # Timebase and frame rate
        assert self.stream.time_base is not None
        self.time_base: fractions.Fraction = self.stream.time_base
        self.frames: int = self.stream.frames
        assert self.stream.average_rate is not None
        self.fps: fractions.Fraction = self.stream.average_rate
        # Duration (in seconds)
        self.duration = float(self.frames) / float(self.fps)
    
    def getFrameAt(self, seconds: float):
        """Seek to the nearest keyframe before seconds and decode next frame."""
        # Convert seconds to pts
        target_pts = int(seconds / float(self.time_base))
        # Seek in container
        self.container.seek(target_pts, any_frame=False, backward=True, stream=self.stream)
        for frame in self.container.decode(self.stream):
            if frame.pts >= target_pts:
                return frame.to_image()
        # fallback to first frame
        frame = next(self.container.decode(self.stream), None)
        if frame:
            return frame.to_image()
        return None

class QueueWriter:
    def __init__(self, queue):
        self.queue = queue
    def write(self, msg):
        if msg:
            self.queue.put(msg)
    def flush(self):
        pass

class MagiaTimelineGUI(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"MagiaTimeline {VERSION} GUI")
        self.geometry("1080x500")
        self.wm_iconbitmap()
        self.iconphoto(False, ImageTk.PhotoImage(file="./logo/MagiaTimeline-Logo-Transparent.png"))
        self.bind("<Destroy>", lambda e: self.onClosing())

        # Layout: Left=video, Right=console
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.player: typing.Optional[VideoPlayer] = None
        self.currentPilImage: typing.Optional[Image.Image] = None
        self.rectId: typing.Optional[int] = None
        self.process: typing.Optional[multiprocessing.Process] = None
        self.queue = multiprocessing.Queue()

        # Left frame
        self.leftFrame = customtkinter.CTkFrame(self)
        self.leftFrame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.leftFrame.grid_columnconfigure(0, weight=1)
        self.leftFrame.grid_rowconfigure(0, weight=1)

        # Video canvas with vertical sliders
        self.videoFrame = customtkinter.CTkFrame(self.leftFrame)
        self.videoFrame.grid(row=0, column=0, sticky="nsew")
        self.videoFrame.grid_columnconfigure(0, weight=1)
        self.videoFrame.grid_columnconfigure(1, weight=0)
        self.videoFrame.grid_columnconfigure(2, weight=0)
        self.videoFrame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.videoFrame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        # Bind resizing to auto-scale image
        self.canvas.bind('<Configure>', self.onCanvasResize)

        self.sliderTop = customtkinter.CTkSlider(self.videoFrame, from_=0, to=1, orientation="vertical", command=self.onSliderChange("top"))
        self.sliderBottom = customtkinter.CTkSlider(self.videoFrame, from_=0, to=1, orientation="vertical", command=self.onSliderChange("bottom"))
        self.sliderTop.grid(row=0, column=1, sticky="ns", padx=(4, 2))
        self.sliderTop.set(1 - 0.75)
        self.sliderBottom.grid(row=0, column=2, sticky="ns", padx=2)
        self.sliderBottom.set(1 - 0.995)

        # Horizontal sliders
        self.horizontalSliderFrame = customtkinter.CTkFrame(self.leftFrame)
        self.horizontalSliderFrame.grid(row=1, column=0, sticky="ews", pady=(2,0))
        self.horizontalSliderFrame.grid_columnconfigure(0, weight=1)
        self.sliderLeft = customtkinter.CTkSlider(self.horizontalSliderFrame, from_=0, to=1, command=self.onSliderChange("left"))
        self.sliderRight = customtkinter.CTkSlider(self.horizontalSliderFrame, from_=0, to=1, command=self.onSliderChange("right"))
        self.sliderLeft.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.sliderLeft.set(0.005)
        self.sliderRight.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        self.sliderRight.set(0.995)

        # Bottom video controls: open, timestamp entry, jump, random
        self.controlFrame = customtkinter.CTkFrame(self.leftFrame)
        self.controlFrame.grid(row=2, column=0, sticky="ews", pady=(10,0))
        self.controlFrame.grid_columnconfigure((0,1,2,3), weight=1)

        # Open video file button
        self.btnOpen = customtkinter.CTkButton(self.controlFrame, text="Open Video", command=self.openVideo)
        self.btnOpen.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        # Timestamp entry
        self.entryTime = customtkinter.CTkEntry(self.controlFrame, placeholder_text="HH:MM:SS.ss")
        self.entryTime.insert(0, "00:00:00.00")
        self.entryTime.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        # Jump to time button
        self.btnJump = customtkinter.CTkButton(self.controlFrame, text="Go To", command=self.jumpToTime)
        self.btnJump.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        # Random time button
        self.btnRandom = customtkinter.CTkButton(self.controlFrame, text="Random", command=self.jumpRandom)
        self.btnRandom.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Right frame: console output and action buttons
        self.rightFrame = customtkinter.CTkFrame(self, width=100)
        self.rightFrame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.rightFrame.grid_rowconfigure(0, weight=1)
        self.rightFrame.grid_rowconfigure(1, weight=0)
        self.rightFrame.grid_rowconfigure(2, weight=0)
        self.rightFrame.grid_columnconfigure(0, weight=1)

        # Console output textbox (read-only)
        self.textbox = customtkinter.CTkTextbox(self.rightFrame)
        self.textbox.grid(row=0, column=0, sticky="nsew", pady=(0,10))
        # make textbox read-only
        self.textbox.configure(state="disabled")

        # Progress bar
        self.progressBar = customtkinter.CTkProgressBar(self.rightFrame, mode="determinate")
        self.progressBar.grid(row=1, column=0, sticky="ew", padx=5, pady=(0,10))
        self.progressBar.set(1.0)
        self.progressBar.stop()

        # Action buttons: Start and Abort
        self.actionFrame = customtkinter.CTkFrame(self.rightFrame)
        self.actionFrame.grid(row=2, column=0, sticky="ew")
        self.actionFrame.grid_columnconfigure(0, weight=1)
        self.actionFrame.grid_columnconfigure(1, weight=1)

        # Abort button
        self.btnAbort = customtkinter.CTkButton(self.actionFrame, text="Abort", fg_color="#ff4d4d", hover_color="#ff1a1a", command=self.abortProcess)
        self.btnAbort.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btnAbort.configure(state="disabled") # Initially disabled
        # Start button
        self.btnStart = customtkinter.CTkButton(self.actionFrame, text="Start", command=self.startProcess)
        self.btnStart.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.after(100, self.consolePoll)

        self.tempDir: typing.Optional[tempfile.TemporaryDirectory] = None

    def writeConsole(self, msg: str):
        """Helper to append text to the read-only console."""
        self.textbox.configure(state="normal")
        self.textbox.insert("end", msg)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def openVideo(self):
        filePath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if filePath:
            self.player = VideoPlayer(filePath)
            img = self.player.getFrameAt(0.0)
            if img:
                self.currentPilImage = img
                self.displayScaledImage()
                self.writeConsole(f"Opened video: {filePath}\n")

    def jumpToTime(self):
        ts = self.entryTime.get()
        parts = ts.split(':')
        try:
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2])
            total = h*3600 + m*60 + s
        except:
            self.writeConsole(f"Invalid timestamp: {ts}\n")
            return
        if self.player:
            img = self.player.getFrameAt(total)
            if img:
                self.currentPilImage = img
                self.displayScaledImage()
                self.writeConsole(f"Jumped to: {ts}\n")
        else:
            self.writeConsole("No video loaded.\n")

    def jumpRandom(self):
        if not self.player:
            return self.writeConsole("No video loaded.\n")
        total = random.uniform(0, self.player.duration)
        img = self.player.getFrameAt(total)
        if img:
            # format hh:mm:ss.ss
            h = int(total // 3600)
            m = int((total % 3600) // 60)
            s = total % 60
            ts_str = f"{h:02d}:{m:02d}:{s:05.2f}"
            # update entry field
            self.entryTime.delete(0, "end")
            self.entryTime.insert(0, ts_str)
            # show frame
            self.currentPilImage = img
            self.displayScaledImage()
            self.writeConsole(f"Jumped random to: {ts_str}\n")

    def displayScaledImage(self):
        if not self.currentPilImage:
            return
        canvasW = self.canvas.winfo_width()
        canvasH = self.canvas.winfo_height()
        if canvasW < 1 or canvasH < 1:
            return
        # stretch to fill
        resized = self.currentPilImage.resize((canvasW, canvasH))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        # draw crop rectangle on top
        self.updateRectangle()

    def onCanvasResize(self, event):
        self.displayScaledImage()

    def onSliderChange(self, slider_id):
        def onSliderChangeImpl(val):
            th = 1 - self.sliderTop.get()
            bh = 1 - self.sliderBottom.get()
            lw = self.sliderLeft.get()
            rw = self.sliderRight.get()

            # enforce bounds
            if lw > rw:
                if slider_id == "left":
                    rw = lw
                    self.sliderRight.set(rw)
                if slider_id == "right":
                    lw = rw
                    self.sliderLeft.set(lw)
            if th > bh:
                if slider_id == "top":
                    bh = th
                    self.sliderBottom.set(1 - bh)
                if slider_id == "bottom":
                    th = bh
                    self.sliderTop.set(1 - th)
            
            self.displayScaledImage()
        return onSliderChangeImpl

    def updateRectangle(self):
        """Draw a hollow red rectangle according to sliders (left≤right, top≤bottom)."""
        if not self.currentPilImage:
            return
        th = 1 - self.sliderTop.get()
        bh = 1 - self.sliderBottom.get()
        lw = self.sliderLeft.get()
        rw = self.sliderRight.get()
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        x1 = lw * W; x2 = rw * W
        y1 = th * H; y2 = bh * H
        # remove old
        if self.rectId:
            self.canvas.delete(self.rectId)
        # draw new
        self.rectId = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline='red', width=4
        )

    @staticmethod
    def processWorker(queue, *args, **kwargs):
        sys.stdout = sys.stderr = QueueWriter(queue)
        import MagiaTimeline
        MagiaTimeline.main(*args, **kwargs)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def consolePollOnce(self):
        try:
            while True:
                msg: str = self.queue.get_nowait()
                self.writeConsole(msg)
                # Update entry field and video frame on messages like:
                # frame 00:01:01.72
                if msg.startswith("frame "):
                    ts_str = msg.split()[1]
                    self.entryTime.configure(state="normal")
                    self.entryTime.delete(0, "end")
                    self.entryTime.insert(0, ts_str)
                    self.entryTime.configure(state="readonly")
                    # Jump to the frame
                    if self.player:
                        img = self.player.getFrameAt(float(ts_str.split(':')[0]) * 3600 + float(ts_str.split(':')[1]) * 60 + float(ts_str.split(':')[2]))
                        if img:
                            self.currentPilImage = img
                            self.displayScaledImage()
        except:
            pass

    def consolePoll(self):
        self.consolePollOnce()
        self.after(100, self.consolePoll)

    @staticmethod
    def startWithHook(p: multiprocessing.Process, on_exit: typing.Callable[[], None]):
        p.start()
        threading.Thread(target=lambda: (p.join(), on_exit()), daemon=True).start()
        return p

    def startProcess(self):
        if self.process and self.process.is_alive():
            return self.writeConsole("Process already running.\n")

        if not self.player:
            return self.writeConsole("No video loaded.\n")
        
        th = 1 - self.sliderTop.get()
        bh = 1 - self.sliderBottom.get()
        lw = self.sliderLeft.get()
        rw = self.sliderRight.get()

        schema = json.load(open("ConfigSchema.json", "r"))
        config = yaml.load(open("config.yml", "r").read(), Loader=yaml.FullLoader)

        config["source"] = [self.player.path]
        config["dtd"]["default"]["dialogRect"] = [lw, rw, th, bh]

        self.writeConsole("Starting process...\n")
        self.tempDir = tempfile.TemporaryDirectory(prefix="MagiaTimeline_")
        self.process = multiprocessing.Process(
            target=MagiaTimelineGUI.processWorker,
            args=(self.queue, config, schema, self.tempDir.name)
        )
        self.startWithHook(
            self.process,
            on_exit=self.enableControls
        )

        self.disableControls()
        self.writeConsole("Process started.\n")

    def disableControls(self):
        self.btnStart.configure(state="disabled")
        self.btnAbort.configure(state="normal")
        self.sliderTop.configure(state="disabled")
        self.sliderBottom.configure(state="disabled")
        self.sliderLeft.configure(state="disabled")
        self.sliderRight.configure(state="disabled")
        self.entryTime.configure(state="disabled")
        self.btnOpen.configure(state="disabled")
        self.btnJump.configure(state="disabled")
        self.btnRandom.configure(state="disabled")
        self.progressBar.configure(mode="indeterminate")
        self.progressBar.start()

    def abortProcess(self):
        if self.process and self.process.is_alive():
            self.writeConsole("Aborting process...\n")
            self.process.terminate()
            self.process.join()
            self.process = None
            self.consolePollOnce()
            self.writeConsole("Process aborted.\n")
            assert self.tempDir is not None
            self.tempDir.cleanup()
            self.tempDir = None
            self.enableControls()
        else:
            self.writeConsole("No process running.\n")

    def enableControls(self):
        self.btnStart.configure(state="normal")
        self.btnAbort.configure(state="disabled")
        self.sliderTop.configure(state="normal")
        self.sliderBottom.configure(state="normal")
        self.sliderLeft.configure(state="normal")
        self.sliderRight.configure(state="normal")
        self.entryTime.configure(state="normal")
        self.btnOpen.configure(state="normal")
        self.btnJump.configure(state="normal")
        self.btnRandom.configure(state="normal")
        self.progressBar.configure(mode="determinate")
        self.progressBar.set(1.0)
        self.progressBar.stop()

    def onClosing(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
            self.process = None
            assert self.tempDir is not None
            self.tempDir.cleanup()
            self.tempDir = None

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows compatibility
    app = MagiaTimelineGUI()
    app.mainloop()
