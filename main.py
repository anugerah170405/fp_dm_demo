import os
import sys
import io
import time
import glob
import json
import shutil
import logging
logging.getLogger().setLevel(logging.ERROR)
import cv2
import pandas as pd
from ultralytics import YOLO
from rich.console import Console
from rich.table import Table
from rich import box

# Cross-platform terminal setup
if os.name == 'nt':
    import msvcrt
    os.system('')  # Enable VT100/ANSI processing in Windows terminal
else:
    import tty
    import termios


class Config:
    """Simple JSON-based configuration management."""
    FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    DEFAULT = {"model": "model/best_e6_int8_openvino_model", "conf": 0.15, "iou": 0.45, "imgsz": 640}

    @classmethod
    def load(cls):
        if os.path.exists(cls.FILE):
            try:
                with open(cls.FILE, "r") as f:
                    return {**cls.DEFAULT, **json.load(f)}
            except Exception:
                pass
        cls.save(cls.DEFAULT)
        return dict(cls.DEFAULT)

    @classmethod
    def save(cls, data):
        with open(cls.FILE, "w") as f:
            json.dump(data, f, indent=2)


class Menu:
    """Minimalist cross-platform CLI table menu selector."""

    @staticmethod
    def get_key():
        if os.name == 'nt':
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'):
                k = msvcrt.getch()
                if k == b'H': return 'up'
                if k == b'P': return 'down'
                return None
            if ch in (b'\r', b'\n', b' '): return 'enter'
            if ch in (b'\x03', b'q', b'Q'): return 'quit'
            try:
                s = ch.decode('utf-8', errors='ignore')
                if s.isdigit(): return s
            except Exception:
                pass
            return None
        else:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                k2 = sys.stdin.read(1)
                if k2 == '[':
                    k3 = sys.stdin.read(1)
                    if k3 == 'A': return 'up'
                    if k3 == 'B': return 'down'
                return None
            if ch in ('\r', '\n', ' '): return 'enter'
            if ch.lower() in ('q', '\x03'): return 'quit'
            if ch.isdigit(): return ch
            return None

    @classmethod
    def select(cls, title, options):
        if not options or not sys.stdin.isatty():
            return 0
        idx, n = 0, len(options)
        fd = None
        old = None
        if os.name != 'nt':
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)

        prev_lines = 0

        def render(first=False):
            nonlocal prev_lines
            table = Table(box=box.ROUNDED)
            table.add_column("#", justify="center", style="cyan", width=3)
            table.add_column(title, style="bold")
            for i, opt in enumerate(options):
                num = str(i + 1)
                line = f"[green]>[/green] [reverse] {opt} [/reverse]" if i == idx else f"  {opt}"
                table.add_row(num, line)

            buf = io.StringIO()
            c = Console(file=buf, force_terminal=True)
            c.print(table)
            c.print(f"[dim](Gunakan ↑/↓ dan Enter, atau angka 1-{n})[/dim]")
            out = buf.getvalue()

            if not first and prev_lines > 0:
                sys.stdout.write(f"\033[{prev_lines}A\r\033[J")
            sys.stdout.write(out)
            sys.stdout.flush()
            prev_lines = len(out.splitlines())

        try:
            render(True)
            while True:
                k = cls.get_key()
                if k == 'up':
                    idx = (idx - 1) % n
                    render()
                elif k == 'down':
                    idx = (idx + 1) % n
                    render()
                elif k == 'enter':
                    break
                elif k == 'quit':
                    idx = n - 1
                    break
                elif k and k.isdigit():
                    val = int(k)
                    if 1 <= val <= n:
                        idx = val - 1
                        break
        finally:
            if old is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print()
        return idx


class Detector:
    """YOLO model wrapper for image and webcam inference."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(self._resolve(cfg["model"]))

    def _resolve(self, path):
        base = os.path.dirname(os.path.abspath(__file__))
        full = os.path.normpath(os.path.join(base, path))
        return full if os.path.exists(full) else path

    def set_params(self, cfg):
        self.cfg = cfg
        self.model = YOLO(self._resolve(cfg["model"]))

    def predict(self, source, imgsz=None):
        t0 = time.perf_counter()
        res = self.model.predict(
            source=source,
            conf=self.cfg["conf"],
            iou=self.cfg["iou"],
            imgsz=imgsz or self.cfg["imgsz"],
            verbose=False
        )[0]
        ms = (time.perf_counter() - t0) * 1000

        boxes = res.boxes
        n = len(boxes)
        confs = [float(b.conf[0]) for b in boxes]
        avg_c = sum(confs) / n if n else 0.0

        if n == 0: status = "Healthy"
        elif n <= 3: status = "Mild"
        elif n <= 8: status = "Moderate"
        else: status = "Severe"

        inf_ms = res.speed.get("inference", 0.0)
        return res, {"total": n, "status": status, "avg_conf": avg_c, "ms": ms, "inf_ms": inf_ms}


class App:
    """Main application orchestrator."""
    MODELS = [
        ("E6 OpenVINO INT8 (Final - Recommended)", "model/best_e6_int8_openvino_model"),
        ("E6 PyTorch (.pt)", "model/best_E6.pt"),
        ("E2 (High-Res 1280)", "model/best_E2.pt"),
        ("E5 (P2-Head)", "model/best_E5.pt"),
        ("E1 (YOLO26s)", "model/best_E1.pt"),
        ("E0 (Baseline)", "model/best_E0.pt")
    ]

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg = Config.load()
        self.detector = Detector(self.cfg)
        self.upload_dir = os.path.join(self.base_dir, "upload")
        self.out_img_dir = os.path.join(self.base_dir, "output", "images")
        self.csv_path = os.path.join(self.base_dir, "output", "laporan_deteksi.csv")
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.out_img_dir, exist_ok=True)

    def run(self):
        # Direct CLI argument support: python Demo/main.py <file/folder>
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            self.process_files([sys.argv[1]])
            return

        while True:
            model_name = os.path.basename(self.cfg["model"])
            status = f"[{model_name} | conf {self.cfg['conf']:.2f} | {self.cfg['imgsz']}px]"
            p = Menu.select(f"FAW DETECTION {status}", [
                "Detect Now",
                "View CSV",
                "Upload",
                "Camera",
                "Settings",
                "Exit"
            ])
            if p == 0: self.do_detect()
            elif p == 1: self.do_view_csv()
            elif p == 2: self.do_upload()
            elif p == 3: self.do_camera()
            elif p == 4: self.do_settings()
            elif p in (5, -1): break

    def do_view_csv(self):
        if not os.path.exists(self.csv_path):
            print(f"\n[!] File laporan belum ditemukan ({os.path.basename(self.csv_path)}).")
            print("    Silakan jalankan deteksi terlebih dahulu.\n")
            Menu.select("Done", ["Back to Main Menu"])
            return

        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                print("\n[!] File laporan CSV kosong.\n")
                Menu.select("Done", ["Back to Main Menu"])
                return

            table = Table(title=f"Laporan Deteksi ({len(df)} item)", box=box.ROUNDED)
            table.add_column("#", justify="center", style="cyan", width=3)
            table.add_column("File", style="bold")
            table.add_column("Status", justify="center")
            table.add_column("Lesions", justify="right")
            table.add_column("Avg Conf", justify="right")
            table.add_column("Inference", justify="right")
            table.add_column("Total Time", justify="right")

            status_styles = {"Healthy": "green", "Mild": "cyan", "Moderate": "yellow", "Severe": "red"}
            for i, row in df.iterrows():
                st = str(row.get("Status", "-"))
                color = status_styles.get(st, "white")
                table.add_row(
                    str(i + 1),
                    str(row.get("File", "-")),
                    f"[{color}]{st}[/{color}]",
                    str(row.get("Total Lesions", "-")),
                    str(row.get("Avg Conf", "-")),
                    f"{row.get('Inference (ms)', '-')} ms",
                    f"{row.get('Total Time (ms)', '-')} ms"
                )

            Console().print(table)
            print()
            Menu.select("Done", ["Back to Main Menu"])
        except Exception as e:
            print(f"\n[!] Gagal membaca CSV: {e}\n")
            Menu.select("Done", ["Back to Main Menu"])

    def do_upload(self):
        path = None
        # 1. Native GUI file dialog (tkinter built-in across Windows & Mac)
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Pilih Gambar Daun Jagung",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
            )
            root.destroy()
        except Exception:
            pass

        # 2. Fallback to macOS osascript if tkinter unavailable
        if not path and sys.platform == "darwin":
            cmd = "osascript -e 'POSIX path of (choose file of type {\"public.image\"} with prompt \"Select Leaf Image:\")' 2>/dev/null"
            path = os.popen(cmd).read().strip()

        # 3. Fallback to terminal input prompt
        if not path:
            print("\n[Input Manual] Masukkan path file gambar (atau tekan Enter untuk batal):")
            path = input("> ").strip().strip('"').strip("'")

        if path and os.path.exists(path):
            dest = os.path.join(self.upload_dir, os.path.basename(path))
            shutil.copy(path, dest)
            print(f"\n[OK] Berhasil mengunggah: {os.path.basename(path)}\n")
        elif path:
            print(f"\n[!] File tidak ditemukan: {path}\n")

    def do_detect(self):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            files.extend(glob.glob(os.path.join(self.upload_dir, ext)))
        files = sorted(files)

        if not files:
            print(f"\n[!] Upload folder is empty. Please upload an image first.\n")
            Menu.select("Back", ["Back to Main Menu"])
            return

        self.process_files(files)
        Menu.select("Done", ["Back to Main Menu"])

    def process_files(self, files):
        console = Console()
        console.print(f"\n[bold blue]Processing {len(files)} image(s)...[/bold blue]")

        table = Table(title="FAW Detection Summary", box=box.ROUNDED)
        table.add_column("#", justify="center", style="dim", width=4)
        table.add_column("File", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Lesions", justify="right")
        table.add_column("Avg Conf", justify="right")
        table.add_column("Inference", justify="right")
        table.add_column("Total Time", justify="right")

        records = []
        status_styles = {"Healthy": "green", "Mild": "cyan", "Moderate": "yellow", "Severe": "red"}

        for i, path in enumerate(files):
            if not os.path.exists(path):
                console.print(f"[red][!] File not found: {path}[/red]")
                continue
            fname = os.path.basename(path)
            res, info = self.detector.predict(path)

            out_path = os.path.join(self.out_img_dir, f"hasil_{fname}")
            cv2.imwrite(out_path, res.plot())

            conf_str = f"{info['avg_conf']:.1%}" if info['total'] else "-"
            st_color = status_styles.get(info["status"], "white")

            table.add_row(
                str(i + 1),
                fname,
                f"[{st_color}]{info['status']}[/{st_color}]",
                str(info["total"]),
                conf_str,
                f"{info['inf_ms']:.1f} ms",
                f"{info['ms']:.1f} ms"
            )

            records.append({
                "File": fname,
                "Status": info["status"],
                "Total Lesions": info["total"],
                "Avg Conf": conf_str,
                "Inference (ms)": round(info["inf_ms"], 1),
                "Total Time (ms)": round(info["ms"], 1),
                "Output": out_path
            })

        console.print(table)
        pd.DataFrame(records).to_csv(self.csv_path, index=False)
        console.print(f"[dim]Saved to:[/dim] [cyan]{self.out_img_dir}/[/cyan] [dim]&[/dim] [cyan]{self.csv_path}[/cyan]\n")

    def do_camera(self):
        print("\n[CAMERA] Opening webcam... Press 'q' on video window to exit.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("[!] Camera unavailable or permission denied.")
            return

        f_count, last_boxes, last_status, prev_t, fps = 0, [], "Scanning...", time.time(), 30.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                f_count += 1
                curr_t = time.time()
                fps = 1.0 / (curr_t - prev_t) if (curr_t - prev_t) > 0 else 30.0
                prev_t = curr_t

                # Run inference every 2 frames for smooth lightweight execution
                if f_count % 2 == 0:
                    res, info = self.detector.predict(frame, imgsz=480)
                    last_boxes = [(*map(int, b.xyxy[0]), float(b.conf[0])) for b in res.boxes]
                    last_status = f"{info['status']} ({info['total']} lesions)"

                for (x1, y1, x2, y2, c) in last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{c:.0%}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                cv2.putText(frame, f"FPS: {fps:.0f} | {last_status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("FAW Scanner (Press 'q' to exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            for _ in range(3): cv2.waitKey(1)

    def do_reset_output(self):
        count = 0
        for f in glob.glob(os.path.join(self.out_img_dir, "*")):
            try:
                os.remove(f)
                count += 1
            except Exception:
                pass
        if os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
            except Exception:
                pass
        print(f"\n[OK] Output berhasil di-reset ({count} gambar & file CSV dihapus).\n")

    def do_settings(self):
        while True:
            m_name = os.path.basename(self.cfg["model"])
            opts = [
                f"Model      : {m_name}",
                f"Threshold  : {self.cfg['conf']:.2f}",
                f"Input Size : {self.cfg['imgsz']} px",
                "Reset Default",
                "Reset Output",
                "Back"
            ]
            p = Menu.select("SETTINGS", opts)
            if p == 0:
                p_m = Menu.select("SELECT MODEL", [m[0] for m in self.MODELS] + ["Cancel"])
                if 0 <= p_m < len(self.MODELS):
                    self.cfg["model"] = self.MODELS[p_m][1]
                    Config.save(self.cfg)
                    self.detector.set_params(self.cfg)
            elif p == 1:
                vals = [0.10, 0.15, 0.20, 0.25, 0.30]
                p_t = Menu.select("SELECT CONFIDENCE", [f"{v:.2f}" for v in vals] + ["Cancel"])
                if 0 <= p_t < len(vals):
                    self.cfg["conf"] = vals[p_t]
                    Config.save(self.cfg)
                    self.detector.set_params(self.cfg)
            elif p == 2:
                sizes = [480, 640, 1280]
                p_s = Menu.select("SELECT INPUT SIZE", [f"{s} px" for s in sizes] + ["Cancel"])
                if 0 <= p_s < len(sizes):
                    self.cfg["imgsz"] = sizes[p_s]
                    Config.save(self.cfg)
                    self.detector.set_params(self.cfg)
            elif p == 3:
                self.cfg = Config.DEFAULT.copy()
                Config.save(self.cfg)
                self.detector.set_params(self.cfg)
                print("\n[OK] Settings reset to default.\n")
            elif p == 4:
                self.do_reset_output()
            elif p in (5, -1):
                break


if __name__ == "__main__":
    App().run()
