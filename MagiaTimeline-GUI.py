import tkinter as tk
from tkinter import filedialog
import customtkinter
from PIL import Image, ImageTk, Image
import typing
import av
import av.container
import av.video
import fractions
import sys
import multiprocessing
import threading
import json
import yaml
import tempfile
import traceback
import queue

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
        self.timeBase: fractions.Fraction = self.stream.time_base
        self.frames: int = self.stream.frames
        assert self.stream.average_rate is not None
        self.fps: fractions.Fraction = self.stream.average_rate
        # Duration (in seconds)
        self.duration = float(self.frames) / float(self.fps)

    def getFrameAt(self, seconds: float):
        """Seek to the nearest keyframe before seconds and decode next frame."""
        seconds = max(0.0, min(seconds, self.duration))
        targetPts = int(seconds / float(self.timeBase))
        try:
            self.container.seek(targetPts, any_frame=False, backward=True, stream=self.stream)
            for frame in self.container.decode(self.stream):
                if frame.pts >= targetPts:
                    return frame.to_image()
            frame = next(self.container.decode(self.stream), None)
            if frame:
                return frame.to_image()
        except Exception:
            pass
        return None

class QueueWriter:
    def __init__(self, queue):
        self.queue = queue
    def write(self, msg):
        if msg:
            self.queue.put(msg)
    def flush(self):
        pass

class FrameSeekService:
    """Coalescing seek service: callers post a time; only the latest pending seek is decoded."""
    def __init__(
        self,
        schedule: typing.Callable,
        getPlayer: typing.Callable[[], typing.Optional[VideoPlayer]],
        onFrame: typing.Callable[[Image.Image], None],
    ):
        self.schedule = schedule
        self.getPlayer = getPlayer
        self.onFrame = onFrame
        self.pending: typing.Optional[float] = None
        self.event = threading.Event()
        threading.Thread(target=self.worker, daemon=True).start()

    def request(self, seconds: float) -> None:
        self.pending = seconds
        self.event.set()

    def worker(self) -> None:
        while True:
            try:
                self.event.wait()
                # clear() before reading pending is intentional: if the main thread
                # posts a new request between clear() and the read below, we pick up
                # the newer value and the re-set event causes one extra (harmless) loop.
                # Reading before clear() would be worse: we could miss a request posted
                # in the clear->read window and stall until the next drag event.
                self.event.clear()
                t = self.pending
                player = self.getPlayer()
                if t is None or player is None:
                    continue
                img = player.getFrameAt(t)
                if img:
                    self.schedule(0, lambda i=img: self.onFrame(i))
            except Exception:
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
        self.rectNorm = {"left": 0.005, "right": 0.995, "top": 0.75, "bottom": 0.995}
        self.dragMode: typing.Optional[str] = None
        self.dragStartCanvas: typing.Optional[typing.Tuple[float, float]] = None
        self.rectAtDragStart: typing.Optional[dict] = None
        self.rectEditable: bool = True
        self.process: typing.Optional[multiprocessing.Process] = None
        self.queue = multiprocessing.Queue()

        self.seekSvc = FrameSeekService(
            schedule=self.after,
            getPlayer=lambda: self.player,
            onFrame=self.applySeekResult,
        )

        # Left frame
        self.leftFrame = customtkinter.CTkFrame(self)
        self.leftFrame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.leftFrame.grid_columnconfigure(0, weight=1)
        self.leftFrame.grid_rowconfigure(0, weight=1)

        # Video canvas with vertical sliders
        self.videoFrame = customtkinter.CTkFrame(self.leftFrame)
        self.videoFrame.grid(row=0, column=0, sticky="nsew")
        self.videoFrame.grid_columnconfigure(0, weight=1)
        self.videoFrame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.videoFrame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind('<Configure>', self.onCanvasResize)
        self.canvas.bind('<ButtonPress-1>', self.onCanvasPress)
        self.canvas.bind('<B1-Motion>', self.onCanvasDrag)
        self.canvas.bind('<ButtonRelease-1>', self.onCanvasRelease)
        self.canvas.bind('<Motion>', self.onCanvasMotion)

        # Bottom video controls: open button + time seek slider + time label
        self.controlFrame = customtkinter.CTkFrame(self.leftFrame)
        self.controlFrame.grid(row=1, column=0, sticky="ews", pady=(10,0))
        self.controlFrame.grid_columnconfigure(0, weight=0)
        self.controlFrame.grid_columnconfigure(1, weight=1)
        self.controlFrame.grid_columnconfigure(2, weight=0)

        self.btnOpen = customtkinter.CTkButton(self.controlFrame, text="Open Video", command=self.openVideo)
        self.btnOpen.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.sliderTime = customtkinter.CTkSlider(self.controlFrame, from_=0, to=1, command=self.onTimeSliderChange, state="disabled")
        self.sliderTime.set(0)
        self.sliderTime.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.labelTime = customtkinter.CTkLabel(self.controlFrame, text="00:00:00.00", width=90, anchor="e")
        self.labelTime.grid(row=0, column=2, padx=(0,32), pady=5, sticky="e")

        # Right frame: console output and action buttons
        self.rightFrame = customtkinter.CTkFrame(self, width=100)
        self.rightFrame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.rightFrame.grid_rowconfigure(0, weight=1)
        self.rightFrame.grid_rowconfigure(1, weight=0)
        self.rightFrame.grid_rowconfigure(2, weight=0)
        self.rightFrame.grid_rowconfigure(3, weight=0)
        self.rightFrame.grid_rowconfigure(4, weight=0)
        self.rightFrame.grid_rowconfigure(5, weight=0)
        self.rightFrame.grid_columnconfigure(0, weight=1)

        # Console output textbox (read-only)
        self.textbox = customtkinter.CTkTextbox(self.rightFrame)
        self.textbox.grid(row=0, column=0, sticky="nsew", pady=(0,10))
        # make textbox read-only
        self.textbox.configure(state="disabled")

        # Checkbox: Enable Text Extraction
        self.checkboxTextExtraction = customtkinter.CTkCheckBox(self.rightFrame, text="Enable Text Extraction")
        self.checkboxTextExtraction.grid(row=1, column=0, sticky="ew", padx=5, pady=(0,10))

        # Checkbox: Enable Style Classification
        self.checkboxStyleClassify = customtkinter.CTkCheckBox(self.rightFrame, text="Enable Style Classification")
        self.checkboxStyleClassify.grid(row=2, column=0, sticky="ew", padx=5, pady=(0,10))

        # Checkbox: Enable Typewriter Subtitle Support
        self.checkboxTypewriter = customtkinter.CTkCheckBox(self.rightFrame, text="Enable Typewriter Subtitle Support")
        self.checkboxTypewriter.grid(row=3, column=0, sticky="ew", padx=5, pady=(0,10))

        # Checkbox: Enable Additional SRT Output
        self.checkboxOutputSrt = customtkinter.CTkCheckBox(self.rightFrame, text="Enable Additional SRT Output")
        self.checkboxOutputSrt.grid(row=4, column=0, sticky="ew", padx=5, pady=(0,10))

        # Progress bar
        self.progressBar = customtkinter.CTkProgressBar(self.rightFrame, mode="determinate")
        self.progressBar.grid(row=5, column=0, sticky="ew", padx=5, pady=(0,10))
        self.progressBar.set(1.0)
        self.progressBar.stop()

        # Action buttons: Start and Abort
        self.actionFrame = customtkinter.CTkFrame(self.rightFrame)
        self.actionFrame.grid(row=6, column=0, sticky="ew")
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

    @staticmethod
    def fmtTime(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    def applySeekResult(self, img: Image.Image) -> None:
        self.currentPilImage = img
        self.displayScaledImage()

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
            self.sliderTime.configure(to=self.player.duration, state="normal")
            self.sliderTime.set(0)
            self.labelTime.configure(text=self.fmtTime(0.0))
            self.seekSvc.request(0.0)
            self.writeConsole(f"[Info] Opened video: {filePath}\n")

    def onTimeSliderChange(self, val: float):
        self.labelTime.configure(text=self.fmtTime(val))
        if self.player:
            self.seekSvc.request(val)

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
        self.drawRect()

    def onCanvasResize(self, event):
        self.displayScaledImage()

    HANDLE_R = 6
    HANDLE_HIT_R = 10
    HANDLE_CURSORS = {
        "TL": "size_nw_se", "BR": "size_nw_se",
        "TR": "size_ne_sw", "BL": "size_ne_sw",
        "TC": "sb_v_double_arrow", "BC": "sb_v_double_arrow",
        "ML": "sb_h_double_arrow", "MR": "sb_h_double_arrow",
        "move": "fleur",
    }

    def drawRect(self) -> None:
        self.canvas.delete("rect_overlay")
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 1 or H < 1:
            return
        l = self.rectNorm["left"] * W
        r = self.rectNorm["right"] * W
        t = self.rectNorm["top"] * H
        b = self.rectNorm["bottom"] * H
        mx = (l + r) / 2
        my = (t + b) / 2
        HR = self.HANDLE_R
        self.canvas.create_rectangle(l - 3, t - 3, r + 3, b + 3, outline="#336699", width=2, tags="rect_overlay")
        self.canvas.create_rectangle(l, t, r, b, outline="#e0aaf2", width=4, tags="rect_overlay")
        self.canvas.create_rectangle(l + 3, t + 3, r - 3, b - 3, outline="#336699", width=2, tags="rect_overlay")
        for hx, hy in [(l, t), (mx, t), (r, t), (l, my), (r, my), (l, b), (mx, b), (r, b)]:
            self.canvas.create_rectangle(
                hx - HR - 4, hy - HR - 4, hx + HR + 4, hy + HR + 4,
                fill="#336699", outline="", tags="rect_overlay"
            )
            self.canvas.create_rectangle(
                hx - HR - 2, hy - HR - 2, hx + HR + 2, hy + HR + 2,
                fill="#e0aaf2", outline="", tags="rect_overlay"
            )
            self.canvas.create_rectangle(
                hx - HR + 2, hy - HR + 2, hx + HR - 2, hy + HR - 2,
                fill="#336699", outline="", tags="rect_overlay"
            )

    def hitTest(self, cx: float, cy: float) -> typing.Optional[str]:
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        l = self.rectNorm["left"] * W
        r = self.rectNorm["right"] * W
        t = self.rectNorm["top"] * H
        b = self.rectNorm["bottom"] * H
        mx = (l + r) / 2
        my = (t + b) / 2
        R = self.HANDLE_HIT_R
        for name, hx, hy in [
            ("TL", l, t), ("TC", mx, t), ("TR", r, t),
            ("ML", l, my), ("MR", r, my),
            ("BL", l, b), ("BC", mx, b), ("BR", r, b),
        ]:
            if abs(cx - hx) <= R and abs(cy - hy) <= R:
                return name
        if l <= cx <= r and t <= cy <= b:
            return "move"
        return None

    def onCanvasPress(self, event) -> None:
        if not self.rectEditable or not self.player:
            return
        mode = self.hitTest(event.x, event.y)
        if mode:
            self.dragMode = mode
            self.dragStartCanvas = (float(event.x), float(event.y))
            self.rectAtDragStart = dict(self.rectNorm)

    def onCanvasDrag(self, event) -> None:
        if not self.dragMode or not self.rectEditable or not self.dragStartCanvas or not self.rectAtDragStart:
            return
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        dx = (event.x - self.dragStartCanvas[0]) / W
        dy = (event.y - self.dragStartCanvas[1]) / H
        r = dict(self.rectAtDragStart)
        MIN_SIZE = 0.01
        if self.dragMode == "move":
            w = r["right"] - r["left"]
            h = r["bottom"] - r["top"]
            newLeft = max(0.0, min(1.0 - w, r["left"] + dx))
            newTop = max(0.0, min(1.0 - h, r["top"] + dy))
            r["left"] = newLeft
            r["right"] = newLeft + w
            r["top"] = newTop
            r["bottom"] = newTop + h
        else:
            if "L" in self.dragMode:
                r["left"] = max(0.0, min(r["right"] - MIN_SIZE, r["left"] + dx))
            if "R" in self.dragMode:
                r["right"] = min(1.0, max(r["left"] + MIN_SIZE, r["right"] + dx))
            if "T" in self.dragMode:
                r["top"] = max(0.0, min(r["bottom"] - MIN_SIZE, r["top"] + dy))
            if "B" in self.dragMode:
                r["bottom"] = min(1.0, max(r["top"] + MIN_SIZE, r["bottom"] + dy))
        self.rectNorm = r
        self.drawRect()

    def onCanvasRelease(self, event) -> None:
        self.dragMode = None
        self.dragStartCanvas = None
        self.rectAtDragStart = None

    def onCanvasMotion(self, event) -> None:
        if not self.rectEditable or not self.player or self.dragMode:
            return
        mode = self.hitTest(event.x, event.y)
        self.canvas.configure(cursor=self.HANDLE_CURSORS.get(mode, "") if mode else "")

    @staticmethod
    def processWorker(queue, *args, **kwargs):
        sys.stdout = sys.stderr = QueueWriter(queue)
        try:
            import MagiaTimeline
            MagiaTimeline.main(*args, **kwargs)
            print(f"[Info] MagiaTimeline worker process finished successfully.")
        except Exception:
            tb = traceback.format_exc()
            print("[Error] Unhandled exception in MagiaTimeline worker process:\n" + tb)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def consolePollOnce(self):
        try:
            while True:
                msg: str = self.queue.get_nowait()
                self.writeConsole(msg)
                # Update slider, label, and video preview on messages like: frame 00:01:01.72
                if msg.startswith("frame "):
                    tsStr = msg.split()[1]
                    parts = tsStr.split(':')
                    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    self.sliderTime.set(seconds)
                    self.labelTime.configure(text=tsStr)
                    self.seekSvc.request(seconds)
        except queue.Empty:
            pass

    def consolePoll(self):
        self.consolePollOnce()
        self.after(100, self.consolePoll)

    @staticmethod
    def startWithHook(p: multiprocessing.Process, onExit: typing.Callable[[], None]):
        p.start()
        threading.Thread(target=lambda: (p.join(), onExit()), daemon=True).start()
        return p

    def startProcess(self):
        if self.process and self.process.is_alive():
            return self.writeConsole("[Error] Process already running.\n")

        if not self.player:
            return self.writeConsole("[Error] No video loaded.\n")

        lw = self.rectNorm["left"]
        rw = self.rectNorm["right"]
        th = self.rectNorm["top"]
        bh = self.rectNorm["bottom"]

        schema = json.load(open("ConfigSchema.json", "r", encoding="utf-8"))
        config = yaml.load(open("config.yml", "r", encoding="utf-8").read(), Loader=yaml.FullLoader)

        config["source"] = [self.player.path]
        config["destination"] = ["..."]
        config["dtd"]["default"]["dialogRect"] = [lw, rw, th, bh]
        extraJobs = []
        if self.checkboxTextExtraction.get():
            extraJobs.append("ocr")
        if self.checkboxStyleClassify.get():
            extraJobs.append("sty")
        config["extraJobs"] = extraJobs
        config["dtd"]["default"]["enableTypewriter"] = bool(self.checkboxTypewriter.get())
        config["outputSrt"] = bool(self.checkboxOutputSrt.get())

        self.writeConsole("[Info] Starting process...\n")
        self.writeConsole(f"[Trace] dialogRect: [{lw:.3f}, {rw:.3f}, {th:.3f}, {bh:.3f}]\n")
        self.tempDir = tempfile.TemporaryDirectory(prefix="MagiaTimeline_")
        self.process = multiprocessing.Process(
            target=MagiaTimelineGUI.processWorker,
            args=(self.queue, config, schema, self.tempDir.name)
        )
        self.startWithHook(
            self.process,
            onExit=self.enableControls
        )

        self.disableControls()
        self.writeConsole("[Info] Process started.\n")

    def disableControls(self):
        self.btnStart.configure(state="disabled")
        self.btnAbort.configure(state="normal")
        self.rectEditable = False
        self.canvas.configure(cursor="")
        self.sliderTime.configure(state="disabled")
        self.btnOpen.configure(state="disabled")
        self.checkboxTextExtraction.configure(state="disabled")
        self.checkboxStyleClassify.configure(state="disabled")
        self.checkboxTypewriter.configure(state="disabled")
        self.checkboxOutputSrt.configure(state="disabled")
        self.progressBar.configure(mode="indeterminate")
        self.progressBar.start()

    def abortProcess(self):
        if self.process and self.process.is_alive():
            self.writeConsole("[Info] Aborting process...\n")
            self.process.terminate()
            self.process.join()
            self.process = None
            self.consolePollOnce()
            self.writeConsole("[Info] Process aborted.\n")
            assert self.tempDir is not None
            self.tempDir.cleanup()
            self.tempDir = None
            self.enableControls()
        else:
            self.writeConsole("[Error] No process running.\n")

    def enableControls(self):
        self.btnStart.configure(state="normal")
        self.btnAbort.configure(state="disabled")
        self.rectEditable = True
        self.sliderTime.configure(state="normal")
        self.btnOpen.configure(state="normal")
        self.checkboxTextExtraction.configure(state="normal")
        self.checkboxStyleClassify.configure(state="normal")
        self.checkboxTypewriter.configure(state="normal")
        self.checkboxOutputSrt.configure(state="normal")
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
