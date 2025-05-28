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
from multiprocessing import Process, Queue
import threading
import json
import yaml

import MagiaTimeline

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
    
    def get_frame_at(self, seconds: float):
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
        self.title(f"MagiaTimeline {MagiaTimeline.VERSION} GUI")
        self.geometry("1100x500")
        self.wm_iconbitmap()
        self.iconphoto(False, ImageTk.PhotoImage(file="./logo/MagiaTimeline-Logo-Transparent.png"))
        self.bind("<Destroy>", lambda e: self.on_closing())

        # Layout: Left=video, Right=console
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.player: typing.Optional[VideoPlayer] = None
        self.current_pil_image: typing.Optional[Image.Image] = None
        self.rect_id: typing.Optional[int] = None
        self.process: typing.Optional[Process] = None
        self.queue = Queue()

        # Left frame
        self.left_frame = customtkinter.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(0, weight=1)

        # Video canvas with vertical sliders
        self.video_frame = customtkinter.CTkFrame(self.left_frame)
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(1, weight=0)
        self.video_frame.grid_columnconfigure(2, weight=0)
        self.video_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        # Bind resizing to auto-scale image
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        self.slider_top = customtkinter.CTkSlider(self.video_frame, from_=0, to=1, orientation="vertical", command=self._on_slider_change("top"))
        self.slider_bottom = customtkinter.CTkSlider(self.video_frame, from_=0, to=1, orientation="vertical", command=self._on_slider_change("bottom"))
        self.slider_top.grid(row=0, column=1, sticky="ns", padx=2)
        self.slider_top.set(1 - 0.75)
        self.slider_bottom.grid(row=0, column=2, sticky="ns", padx=2)
        self.slider_bottom.set(1 - 0.99)

        # Horizontal sliders
        self.horizontal_slider_frame = customtkinter.CTkFrame(self.left_frame)
        self.horizontal_slider_frame.grid(row=1, column=0, sticky="ews", pady=(10,0))
        self.horizontal_slider_frame.grid_columnconfigure(0, weight=1)
        self.slider_left = customtkinter.CTkSlider(self.horizontal_slider_frame, from_=0, to=1, command=self._on_slider_change("left"))
        self.slider_right = customtkinter.CTkSlider(self.horizontal_slider_frame, from_=0, to=1, command=self._on_slider_change("right"))
        self.slider_left.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
        self.slider_left.set(0.01)
        self.slider_right.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        self.slider_right.set(0.99)

        # Bottom video controls: open, timestamp entry, jump, random
        self.control_frame = customtkinter.CTkFrame(self.left_frame)
        self.control_frame.grid(row=2, column=0, sticky="ews", pady=(10,0))
        self.control_frame.grid_columnconfigure((0,1,2,3), weight=1)

        # Open video file button
        self.btn_open = customtkinter.CTkButton(self.control_frame, text="Open Video", command=self.open_video)
        self.btn_open.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        # Timestamp entry
        self.entry_time = customtkinter.CTkEntry(self.control_frame, placeholder_text="HH:MM:SS.ss")
        self.entry_time.insert(0, "00:00:00.00")
        self.entry_time.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        # Jump to time button
        self.btn_jump = customtkinter.CTkButton(self.control_frame, text="Go To", command=self.jump_to_time)
        self.btn_jump.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        # Random time button
        self.btn_random = customtkinter.CTkButton(self.control_frame, text="Random", command=self.jump_random)
        self.btn_random.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Right frame: console output and action buttons
        self.right_frame = customtkinter.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=0)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Console output textbox (read-only)
        self.textbox = customtkinter.CTkTextbox(self.right_frame)
        self.textbox.grid(row=0, column=0, sticky="nsew", pady=(0,10))
        # make textbox read-only
        self.textbox.configure(state="disabled")

        # Action buttons: Start and Abort
        self.action_frame = customtkinter.CTkFrame(self.right_frame)
        self.action_frame.grid(row=1, column=0, sticky="ew")
        self.action_frame.grid_columnconfigure(0, weight=1)
        self.action_frame.grid_columnconfigure(1, weight=1)

        # Abort button
        self.btn_abort = customtkinter.CTkButton(self.action_frame, text="Abort", fg_color="#ff4d4d", hover_color="#ff1a1a", command=self.abort_process)
        self.btn_abort.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.btn_abort.configure(state="disabled") # Initially disabled
        # Start button
        self.btn_start = customtkinter.CTkButton(self.action_frame, text="Start", command=self.start_process)
        self.btn_start.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.after(100, self.gui_poll)

    def write_console(self, msg: str):
        """Helper to append text to the read-only console."""
        self.textbox.configure(state="normal")
        self.textbox.insert("end", msg)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if file_path:
            self.player = VideoPlayer(file_path)
            img = self.player.get_frame_at(0.0)
            if img:
                self.current_pil_image = img
                self._display_scaled_image()
                self.write_console(f"Opened video: {file_path}\n")

    def jump_to_time(self):
        ts = self.entry_time.get()
        parts = ts.split(':')
        try:
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2])
            total = h*3600 + m*60 + s
        except:
            self.write_console(f"Invalid timestamp: {ts}\n")
            return
        if self.player:
            img = self.player.get_frame_at(total)
            if img:
                self.current_pil_image = img
                self._display_scaled_image()
                self.write_console(f"Jumped to: {ts}\n")
        else:
            self.write_console("No video loaded.\n")

    def jump_random(self):
        if not self.player:
            return self.write_console("No video loaded.\n")
        total = random.uniform(0, self.player.duration)
        img = self.player.get_frame_at(total)
        if img:
            # format hh:mm:ss.ss
            h = int(total // 3600)
            m = int((total % 3600) // 60)
            s = total % 60
            ts_str = f"{h:02d}:{m:02d}:{s:05.2f}"
            # update entry field
            self.entry_time.delete(0, "end")
            self.entry_time.insert(0, ts_str)
            # show frame
            self.current_pil_image = img
            self._display_scaled_image()
            self.write_console(f"Jumped random to: {ts_str}\n")

    def _display_scaled_image(self):
        if not self.current_pil_image:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 1 or canvas_h < 1:
            return
        # stretch to fill
        resized = self.current_pil_image.resize((canvas_w, canvas_h))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)
        # draw crop rectangle on top
        self.update_rectangle()

    def _on_canvas_resize(self, event):
        self._display_scaled_image()

    def _on_slider_change(self, slider_id):
        def _on_slider_change_impl(val):
            th = 1 - self.slider_top.get()
            bh = 1 - self.slider_bottom.get()
            lw = self.slider_left.get()
            rw = self.slider_right.get()

            # enforce bounds
            if lw > rw:
                if slider_id == "left":
                    rw = lw
                    self.slider_right.set(rw)
                if slider_id == "right":
                    lw = rw
                    self.slider_left.set(lw)
            if th > bh:
                if slider_id == "top":
                    bh = th
                    self.slider_bottom.set(1 - bh)
                if slider_id == "bottom":
                    th = bh
                    self.slider_top.set(1 - th)
            
            self._display_scaled_image()
        return _on_slider_change_impl

    def update_rectangle(self):
        """Draw a hollow red rectangle according to sliders (left≤right, top≤bottom)."""
        if not self.current_pil_image:
            return
        th = 1 - self.slider_top.get()
        bh = 1 - self.slider_bottom.get()
        lw = self.slider_left.get()
        rw = self.slider_right.get()
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        x1 = lw * W; x2 = rw * W
        y1 = th * H; y2 = bh * H
        # remove old
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        # draw new
        self.rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline='red', width=4
        )

    @staticmethod
    def process_worker(queue, *args, **kwargs):
        sys.stdout = sys.stderr = QueueWriter(queue)
        MagiaTimeline.main(*args, **kwargs)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def gui_poll(self):
        try:
            while True:
                msg: str = self.queue.get_nowait()
                self.write_console(msg)
                # Update entry field and video frame on messages like:
                # frame 00:01:01.72
                if msg.startswith("frame "):
                    ts_str = msg.split()[1]
                    self.entry_time.configure(state="normal")
                    self.entry_time.delete(0, "end")
                    self.entry_time.insert(0, ts_str)
                    self.entry_time.configure(state="readonly")
                    # Jump to the frame
                    if self.player:
                        img = self.player.get_frame_at(float(ts_str.split(':')[0]) * 3600 + float(ts_str.split(':')[1]) * 60 + float(ts_str.split(':')[2]))
                        if img:
                            self.current_pil_image = img
                            self._display_scaled_image()
        except:
            pass
        self.after(100, self.gui_poll)

    @staticmethod
    def start_with_hook(p: Process, on_exit: typing.Callable[[], None]):
        p.start()
        threading.Thread(target=lambda: (p.join(), on_exit()), daemon=True).start()
        return p

    def start_process(self):
        if self.process and self.process.is_alive():
            return self.write_console("Process already running.\n")

        if not self.player:
            return self.write_console("No video loaded.\n")
        
        th = 1 - self.slider_top.get()
        bh = 1 - self.slider_bottom.get()
        lw = self.slider_left.get()
        rw = self.slider_right.get()

        schema = json.load(open("ConfigSchema.json", "r"))
        config = yaml.load(open("config.yml", "r").read(), Loader=yaml.FullLoader)

        config["source"] = [self.player.path]
        config["dtd"]["default"]["dialogRect"] = [lw, rw, th, bh]

        self.write_console("Starting process...\n")
        self.process = Process(
            target=MagiaTimelineGUI.process_worker,
            args=(self.queue, config, schema)
        )
        self.start_with_hook(
            self.process,
            on_exit=self.enable_controls
        )

        self.disable_controls()
        self.write_console("Process started.\n")

    def disable_controls(self):
        self.btn_start.configure(state="disabled")
        self.btn_abort.configure(state="normal")
        self.slider_top.configure(state="disabled")
        self.slider_bottom.configure(state="disabled")
        self.slider_left.configure(state="disabled")
        self.slider_right.configure(state="disabled")
        self.entry_time.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.btn_jump.configure(state="disabled")
        self.btn_random.configure(state="disabled")

    def abort_process(self):
        if self.process and self.process.is_alive():
            self.write_console("Aborting process...\n")
            self.process.terminate()
            self.process.join()
            self.process = None
            self.write_console("Process aborted.\n")
            self.enable_controls()
        else:
            self.write_console("No process running.\n")

    def enable_controls(self):
        self.btn_start.configure(state="normal")
        self.btn_abort.configure(state="disabled")
        self.slider_top.configure(state="normal")
        self.slider_bottom.configure(state="normal")
        self.slider_left.configure(state="normal")
        self.slider_right.configure(state="normal")
        self.entry_time.configure(state="normal")
        self.btn_open.configure(state="normal")
        self.btn_jump.configure(state="normal")
        self.btn_random.configure(state="normal")

    def on_closing(self):
        if self.process and self.process.is_alive():
            self.abort_process()

if __name__ == "__main__":
    app = MagiaTimelineGUI()
    app.mainloop()
