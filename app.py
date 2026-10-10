"""Food Detection and Calorie Estimation - Tkinter + YOLO desktop app.

Run with:  python app.py
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from calorie_data import summarize

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODEL = os.path.join(APP_DIR, "models", "best.pt")
DEFAULT_MODEL = "yolo11n.pt"  # downloaded automatically on first use
CONFIDENCE = 0.25
PREVIEW_SIZE = (640, 480)
IMAGE_TYPES = [("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]


class FoodCalorieApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Detection and Calorie Estimation")
        self.root.geometry("1100x620")
        self.root.minsize(900, 520)

        self.image = None        # PIL RGB image currently loaded
        self.image_path = None
        self.model = None        # loaded lazily on first detection
        self.model_path = CUSTOM_MODEL if os.path.exists(CUSTOM_MODEL) else DEFAULT_MODEL
        self.preview_photo = None  # keep a reference so Tk doesn't garbage-collect it
        self.results = queue.Queue()  # worker thread -> UI thread

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        self.load_btn = ttk.Button(toolbar, text="Load Food Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=4)
        self.detect_btn = ttk.Button(toolbar, text="Detect Food", command=self.detect)
        self.detect_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Choose Model...", command=self.choose_model).pack(side=tk.LEFT, padx=4)

        self.model_var = tk.StringVar(value=f"Model: {os.path.basename(self.model_path)}")
        ttk.Label(toolbar, textvariable=self.model_var).pack(side=tk.RIGHT, padx=4)

        self.status_var = tk.StringVar(value="Load an image to begin.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, padding=4).pack(side=tk.BOTTOM, fill=tk.X)

        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)

        # Left: image preview
        # A fixed-size frame holds the preview; a text-only Label would size in characters.
        holder = tk.Frame(body, bg="#222", width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        holder.pack_propagate(False)
        self.preview = tk.Label(holder, text="No image loaded", bg="#222", fg="#ccc")
        self.preview.pack(fill=tk.BOTH, expand=True)

        # Right: results table and total
        right = ttk.Frame(body, padding=(8, 0, 0, 0))
        right.pack(side=tk.RIGHT, fill=tk.Y)

        columns = ("item", "conf", "portion", "per100", "kcal")
        self.table = ttk.Treeview(right, columns=columns, show="headings", height=18)
        headings = {"item": ("Item", 110), "conf": ("Confidence", 80),
                    "portion": ("Portion (g)", 80), "per100": ("kcal/100 g", 80),
                    "kcal": ("Calories", 70)}
        for col, (text, width) in headings.items():
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor=tk.CENTER)
        self.table.pack(fill=tk.Y, expand=True)

        self.total_var = tk.StringVar(value="Total Calories: --")
        ttk.Label(right, textvariable=self.total_var, font=("Segoe UI", 14, "bold")).pack(pady=(8, 2))
        ttk.Label(right, text="Estimates only - not medical or dietary advice.",
                  foreground="#888").pack()

    def show_image(self, img):
        fitted = img.copy()
        fitted.thumbnail(PREVIEW_SIZE)
        self.preview_photo = ImageTk.PhotoImage(fitted)
        self.preview.configure(image=self.preview_photo, text="")

    def clear_results(self):
        self.table.delete(*self.table.get_children())
        self.total_var.set("Total Calories: --")

    # ---------- Actions ----------
    def load_image(self):
        path = filedialog.askopenfilename(title="Select a food image", filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            with Image.open(path) as img:
                self.image = img.convert("RGB")
        except Exception as exc:
            messagebox.showerror("Invalid image", f"Could not open this image:\n{exc}")
            return
        self.image_path = path
        self.clear_results()
        self.show_image(self.image)
        self.status_var.set(f"Loaded {os.path.basename(path)} - click Detect Food.")

    def choose_model(self):
        path = filedialog.askopenfilename(title="Select YOLO weights",
                                          filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_path = path
            self.model = None  # reload on next detection
            self.model_var.set(f"Model: {os.path.basename(path)}")

    def detect(self):
        if self.image is None:
            messagebox.showwarning("No image", "Please load a food image first.")
            return
        self.detect_btn.state(["disabled"])
        self.load_btn.state(["disabled"])
        self.status_var.set("Detecting... (the first run may download YOLO weights)")
        # Inference runs on a worker thread so the window stays responsive.
        threading.Thread(target=self._detect_worker, args=(self.image,), daemon=True).start()
        self.root.after(100, self._poll_results)

    def _poll_results(self):
        # Tk widgets must only be touched from the main thread, so the worker
        # puts its outcome on a queue and we check it here.
        try:
            handler, *args = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return
        handler(*args)

    def _detect_worker(self, image):
        try:
            if self.model is None:
                from ultralytics import YOLO  # imported lazily: torch is slow to load
                self.model = YOLO(self.model_path)
            # Pass the PIL image directly; Ultralytics handles RGB->BGR itself.
            result = self.model.predict(source=image, conf=CONFIDENCE, verbose=False)[0]
            names = result.names
            detections = [(names[int(c)], float(p))
                          for c, p in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())]
            annotated = Image.fromarray(result.plot()[..., ::-1])  # BGR -> RGB
            self.results.put((self._show_results, detections, annotated))
        except ImportError as exc:
            self.results.put((self._show_error, "Missing package",
                              f"{exc}\n\nActivate the venv and run:\npython -m pip install -r requirements.txt"))
        except Exception as exc:
            self.results.put((self._show_error, "Detection failed", str(exc)))

    def _show_results(self, detections, annotated):
        self.show_image(annotated)
        self.clear_results()
        rows, total = summarize(detections)
        for row in rows:
            self.table.insert("", tk.END, values=(
                row["label"],
                f"{row['confidence']:.0%}",
                row["portion"] if row["portion"] is not None else "-",
                row["per_100g"] if row["per_100g"] is not None else "-",
                row["kcal"] if row["kcal"] is not None else "N/A",
            ))
        matched = sum(1 for r in rows if r["kcal"] is not None)
        self.total_var.set(f"Total Calories: {total:g} kcal")
        if not rows:
            self.status_var.set("No objects detected. Try another image or a food-trained model.")
        elif matched == 0:
            self.status_var.set(f"{len(rows)} object(s) found, none in the calorie database - total is not reliable.")
        else:
            self.status_var.set(f"{len(rows)} object(s) found, {matched} matched to the calorie database.")
        self._enable_buttons()

    def _show_error(self, title, message):
        self.status_var.set(title)
        messagebox.showerror(title, message)
        self._enable_buttons()

    def _enable_buttons(self):
        self.detect_btn.state(["!disabled"])
        self.load_btn.state(["!disabled"])


def main():
    root = tk.Tk()
    FoodCalorieApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
