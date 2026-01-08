import sys
import os
import json
import numpy as np
import sounddevice as sd
import matplotlib
from datetime import datetime
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit,
                             QFileDialog, QGroupBox, QSpinBox, QMessageBox, 
                             QProgressBar, QSlider, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

v2 = len(sys.argv) > 1 and (sys.argv[1] == '--v2')
ONNX_DIR = "assets/onnx" if v2 else "v1_assets/onnx"
VOICES_DIR = "assets/voice_styles" if v2 else "v1_assets/voice_styles"

# Import from the provided helper.py
try:
    if v2:
        from helper import load_text_to_speech, Style
    else:
        from v1_helper import load_text_to_speech, Style
except ImportError:

    ("Error: helper.py not found. Please place it in the same directory.")
    sys.exit(1)

# --- Custom Interactive Heatmap Widget (Fixed Types) ---
class ClickableHeatmap(QWidget):
    data_changed = pyqtSignal() 

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.data = None
        self.title = title

        # Matplotlib Setup
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.layout.addWidget(self.canvas)

        self.canvas.mpl_connect('button_press_event', self.on_click)

    def set_data(self, data):
        # Ensure we start with float32
        self.data = data.copy().astype(np.float32)
        self.redraw()
        self.data_changed.emit()

    def get_data(self):
        return self.data

    def redraw(self):
        self.ax.clear()
        if self.data is not None:
            cax = self.ax.imshow(self.data, cmap='viridis', aspect='auto', interpolation='nearest', origin='upper')
            self.ax.set_title(f"{self.title} {self.data.shape}")
            self.ax.set_xlabel("Features (Left-Click: Shift Right)")
            self.ax.set_ylabel("Time/Tokens (Right-Click: Shift Down)")
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or self.data is None:
            return

        if event.button == 1:  # Left Click -> Shift Columns Right
            self.data = np.roll(self.data, 1, axis=1)
            print("Shifted Features +1")
        elif event.button == 3:  # Right Click -> Shift Rows Down
            self.data = np.roll(self.data, 1, axis=0)
            print("Shifted Time +1")

        self.redraw()
        self.data_changed.emit()

    # --- Basic Ops (Fixed casting) ---
    def mirror_x(self):
        if self.data is not None:
            self.data = np.fliplr(self.data)
            self.redraw(); self.data_changed.emit()

    def mirror_y(self):
        if self.data is not None:
            self.data = np.flipud(self.data)
            self.redraw(); self.data_changed.emit()

    def invert_sign(self):
        if self.data is not None:
            self.data = -self.data
            self.redraw(); self.data_changed.emit()

    def add_scalar(self, val):
        if self.data is not None:
            # Force float32 after addition
            self.data = (self.data + val).astype(np.float32)
            self.redraw(); self.data_changed.emit()

    def multiply_scalar(self, val):
        if self.data is not None:
            # Force float32 after multiplication
            self.data = (self.data * val).astype(np.float32)
            self.redraw(); self.data_changed.emit()

    # --- Calculus (Fixed casting) ---
    def take_derivative(self):
        if self.data is not None:
            # Gradient can return float64
            self.data = np.gradient(self.data, axis=0)
            self.data = self.data.astype(np.float32)
            self.redraw(); self.data_changed.emit()

    # --- NEW RANDOM SHIFT (Replaces Generator) ---
    def randomize_shifts(self):
        """
        Simulates clicking the graph a random number of times.
        Shifts columns (features) and rows (time) by random amounts
        scaled to the dimensions of the data.
        """
        if self.data is None:
            return

        rows, cols = self.data.shape

        # Random shift amounts (0 to size-1 covers all unique positions)
        shift_features = np.random.randint(0, cols)
        shift_time = np.random.randint(0, rows)

        # Apply shifts (np.roll wraps around, equivalent to repeated clicking)
        # Axis 1 = Features (Left/Right)
        self.data = np.roll(self.data, shift_features, axis=1)
        # Axis 0 = Time (Up/Down)
        self.data = np.roll(self.data, shift_time, axis=0)

        print(f"Random Shift: Features +{shift_features}, Time +{shift_time}")
        self.redraw()
        self.data_changed.emit()

    # --- NEW DSP OPERATIONS (Modified to work on Rows/Features) ---
    def dsp_sharpen(self):
        """Adds the derivative across features (axis 1) to exaggerate spectral peaks."""
        if self.data is not None:
            # Changed axis from 0 to 1
            grad = np.gradient(self.data, axis=1)
            # Ensure calculation stays in float32
            self.data = (self.data + (grad * 1.5)).astype(np.float32)
            self.redraw(); self.data_changed.emit()
            print("Applied Sharpen (Features)")

    def dsp_quantize(self):
        """Rounds values to create steps (Bitcrush effect). Axis independent."""
        if self.data is not None:
            factor = 5.0 
            self.data = (np.round(self.data * factor) / factor).astype(np.float32)
            self.redraw(); self.data_changed.emit()
            print("Applied Quantize")

    def dsp_echo(self):
        """Adds a shifted copy of the signal across features (Spectral Smear)."""
        if self.data is not None:
            # Changed axis from 0 to 1
            echo = np.roll(self.data, 2, axis=1) * 0.5
            self.data = (self.data + echo).astype(np.float32)
            self.redraw(); self.data_changed.emit()
            print("Applied Echo (Features)")

    def dsp_tremolo(self):
        """Applies a sine wave amplitude modulation across features (Spectral Ripple)."""
        if self.data is not None:
            rows, cols = self.data.shape
            # Changed linspace to use cols, and broadcasting to [newaxis, :]
            t = np.linspace(0, 2 * np.pi, cols)
            envelope = 1.0 + 0.5 * np.sin(t)
            # Cast result back to float32
            self.data = (self.data * envelope[np.newaxis, :]).astype(np.float32)
            self.redraw(); self.data_changed.emit()
            print("Applied Tremolo (Features)")

    def dsp_jitter(self):
        """Multiplies by random factors. Axis independent."""
        if self.data is not None:
            # np.random returns float64
            jitter_mat = np.random.uniform(0.8, 1.2, self.data.shape)
            self.data = (self.data * jitter_mat).astype(np.float32)
            self.redraw(); self.data_changed.emit()
            print("Applied Jitter")

# --- Worker Thread ---
class InferenceThread(QThread):
    finished = pyqtSignal(object, int)
    error = pyqtSignal(str)

    def __init__(self, tts_engine, text, style, steps, speed):
        super().__init__()
        self.tts_engine = tts_engine
        self.text = text
        self.style = style
        self.steps = steps
        self.speed = speed

    def run(self):
        try:
            params = {
                'text': self.text,
                'style': self.style,
                'total_step': self.steps,
                'speed': self.speed
            }
            if v2:
                params['lang'] = 'en'
            wav, duration = self.tts_engine(**params)
            audio_data = wav.squeeze()
            sample_rate = self.tts_engine.sample_rate
            self.finished.emit(audio_data, sample_rate)
        except Exception as e:
            self.error.emit(str(e))

# --- Main GUI Application ---
class LatentExplorerDSP(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Supertonic Latent Explorer: DSP EDITION (Row/Feature Mode)")
        self.resize(1100, 950)

        self.tts_engine = None
        self.original_ttl = None
        self.original_dp = None 

        # Store for multi-voice mixing
        self.voice_library = [] 

        self.init_ui()

        self.status_bar.showMessage("Loading ONNX models...")
        QThread.msleep(100)
        self.load_model()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- 1. Header (Single File) ---
        header_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Voice JSON")
        self.btn_load.clicked.connect(self.load_voice_json)
        self.btn_save = QPushButton("Save Voice JSON")
        self.btn_save.setStyleSheet("font-weight: bold;")
        self.btn_save.clicked.connect(self.save_voice_json)
        self.btn_save.setEnabled(False)
        self.btn_reset = QPushButton("Reset to Original")
        self.btn_reset.setStyleSheet("font-weight: bold;")
        self.btn_reset.clicked.connect(self.reset_voice)
        self.btn_reset.setEnabled(False)
        self.lbl_info = QLabel("No voice loaded")
        self.lbl_info.setStyleSheet("font-weight: bold; margin-left: 10px;")

        header_layout.addWidget(self.btn_load)
        header_layout.addWidget(self.btn_save)
        header_layout.addWidget(self.btn_reset)
        header_layout.addWidget(self.lbl_info)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # --- 2. Multi-Voice Mixer (New) ---
        mixer_group = QGroupBox("Multi-Voice Mixer")
        mixer_layout = QHBoxLayout()

        self.btn_load_lib = QPushButton("Load Library (2+ files)")
        self.btn_load_lib.setToolTip("Select multiple JSON files to create a mix")
        self.btn_load_lib.clicked.connect(self.load_voice_library)

        self.lbl_lib_status = QLabel("0 voices loaded")

        self.btn_remix = QPushButton("Remix Library")
        self.btn_remix.setToolTip("Mix loaded voices with random weights (Sum=1)")
        self.btn_remix.setStyleSheet("font-weight: bold;")
        self.btn_remix.clicked.connect(self.remix_voices)
        self.btn_remix.setEnabled(False)
        self.weights_input = QLineEdit(self)
        self.weights_input.setPlaceholderText("Mix weights, e.g: 0.85 0.15")
        self.lbl_weights = QLabel("")

        mixer_layout.addWidget(self.btn_load_lib)
        mixer_layout.addWidget(self.lbl_lib_status)
        mixer_layout.addWidget(self.btn_remix)
        mixer_layout.addWidget(self.weights_input)
        mixer_layout.addWidget(self.lbl_weights)
        mixer_layout.addStretch()
        mixer_group.setLayout(mixer_layout)
        main_layout.addWidget(mixer_group)

        # --- 3. Visualizer ---
        self.heatmap_ttl = ClickableHeatmap("Timbre (Style TTL)")
        self.heatmap_ttl.data_changed.connect(self.update_stats)
        main_layout.addWidget(self.heatmap_ttl, stretch=1)

        # --- 4. Operations Group ---
        ops_group = QGroupBox("Latent Operations")
        ops_layout = QVBoxLayout()

        # Row 1: Geometric & Calculus
        row1 = QHBoxLayout()
        btn_mir_x = QPushButton("Mirror X")
        btn_mir_x.clicked.connect(self.heatmap_ttl.mirror_x)
        btn_mir_y = QPushButton("Mirror Y")
        btn_mir_y.clicked.connect(self.heatmap_ttl.mirror_y)
        btn_inv = QPushButton("Invert Sign")
        btn_inv.clicked.connect(self.heatmap_ttl.invert_sign)
        btn_deriv = QPushButton("Derivative")
        btn_deriv.clicked.connect(self.heatmap_ttl.take_derivative)

        # REPLACED GENERATOR WITH RANDOM SHIFT
        btn_rnd_shift = QPushButton("Rand Shift")
        btn_rnd_shift.setToolTip("Randomly shifts features and time (simulates random clicks)")
        btn_rnd_shift.setStyleSheet("font-weight: bold;")
        btn_rnd_shift.clicked.connect(self.heatmap_ttl.randomize_shifts)

        row1.addWidget(btn_mir_x); row1.addWidget(btn_mir_y); row1.addWidget(btn_inv)
        row1.addWidget(btn_deriv); row1.addWidget(btn_rnd_shift)
        ops_layout.addLayout(row1)

        # Row 2: DSP (Signal Processing)
        row2 = QHBoxLayout()
        btn_sharpen = QPushButton("Sharpen")
        btn_sharpen.setToolTip("Adds derivative to signal to exaggerate zigzags")
        btn_sharpen.setStyleSheet("font-weight: bold;")
        btn_sharpen.clicked.connect(self.heatmap_ttl.dsp_sharpen)

        btn_quant = QPushButton("Quantize")
        btn_quant.setToolTip("Rounds values to steps (Bitcrush)")
        btn_quant.setStyleSheet("font-weight: bold;")
        btn_quant.clicked.connect(self.heatmap_ttl.dsp_quantize)

        btn_echo = QPushButton("Echo")
        btn_echo.setToolTip("Adds delayed signal")
        btn_echo.setStyleSheet("font-weight: bold;")
        btn_echo.clicked.connect(self.heatmap_ttl.dsp_echo)

        btn_trem = QPushButton("Tremolo")
        btn_trem.setToolTip("Sine wave amplitude modulation")
        btn_trem.setStyleSheet("font-weight: bold;")
        btn_trem.clicked.connect(self.heatmap_ttl.dsp_tremolo)

        btn_jit = QPushButton("Jitter")
        btn_jit.setToolTip("Random amplitude scaling")
        btn_jit.setStyleSheet("font-weight: bold;")
        btn_jit.clicked.connect(self.heatmap_ttl.dsp_jitter)

        row2.addWidget(btn_sharpen); row2.addWidget(btn_quant); row2.addWidget(btn_echo)
        row2.addWidget(btn_trem); row2.addWidget(btn_jit)
        ops_layout.addLayout(row2)

        # Row 3: Math Ops
        row3 = QHBoxLayout()
        self.spin_add = QDoubleSpinBox()
        self.spin_add.setRange(-10.0, 10.0); self.spin_add.setSingleStep(0.01); self.spin_add.setValue(0.05)
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(lambda: self.heatmap_ttl.add_scalar(self.spin_add.value()))

        self.spin_mul = QDoubleSpinBox()
        self.spin_mul.setRange(-10.0, 10.0); self.spin_mul.setSingleStep(0.1); self.spin_mul.setValue(1.1)
        btn_mul = QPushButton("Multiply")
        btn_mul.clicked.connect(lambda: self.heatmap_ttl.multiply_scalar(self.spin_mul.value()))

        self.lbl_stats = QLabel("Min: N/A | Max: N/A")
        self.lbl_stats.setStyleSheet("font-family: monospace; font-weight: bold;")

        row3.addWidget(QLabel("Val:")); row3.addWidget(self.spin_add); row3.addWidget(btn_add)
        row3.addSpacing(20)
        row3.addWidget(QLabel("Factor:")); row3.addWidget(self.spin_mul); row3.addWidget(btn_mul)
        row3.addStretch(); row3.addWidget(self.lbl_stats)

        ops_layout.addLayout(row3)
        ops_group.setLayout(ops_layout)
        main_layout.addWidget(ops_group)

        # --- 5. Inference Settings ---
        controls_group = QGroupBox("Inference Settings")
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Speed:"))
        self.sl_speed = QSlider(Qt.Horizontal)
        self.sl_speed.setRange(50, 200); self.sl_speed.setValue(135)
        self.lbl_speed = QLabel("1.35x")
        self.sl_speed.valueChanged.connect(lambda v: self.lbl_speed.setText(f"{v/100.0:.2f}x"))
        ctrl_layout.addWidget(self.sl_speed); ctrl_layout.addWidget(self.lbl_speed)
        ctrl_layout.addWidget(QLabel("Steps:"))
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1, 50); self.spin_steps.setValue(10)
        ctrl_layout.addWidget(self.spin_steps)
        controls_group.setLayout(ctrl_layout)
        main_layout.addWidget(controls_group)

        # --- 6. Text & Play ---
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to synthesize...")
        self.text_input.setText("They have a whole ritual for getting in sync with someone to ensure you are someone trustworthy.")
        self.text_input.setMaximumHeight(60)
        main_layout.addWidget(self.text_input)

        self.btn_generate = QPushButton("Generate & Play")
        self.btn_generate.setMinimumHeight(50)
        self.btn_generate.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #4C8F50; color: white;")
        self.btn_generate.clicked.connect(self.start_inference)
        self.btn_generate.setEnabled(False)
        main_layout.addWidget(self.btn_generate)

        # Status Bar
        self.status_bar = self.statusBar()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress)

    def update_stats(self):
        data = self.heatmap_ttl.get_data()
        if data is not None:
            mn = np.min(data)
            mx = np.max(data)
            self.lbl_stats.setText(f"Min: {mn:.4f} | Max: {mx:.4f}")

    def load_model(self):
        if not os.path.exists(ONNX_DIR):
            QMessageBox.critical(self, "Error", f"Model directory not found at: {os.path.abspath(ONNX_DIR)}")
            return
        try:
            self.tts_engine = load_text_to_speech(ONNX_DIR, use_gpu=False)
            self.status_bar.showMessage("Model loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))

    def load_voice_json(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Voice Style", VOICES_DIR, "JSON Files (*.json)")
        if not fname:
            return
        try:
            with open(fname, 'r') as f:
                data = json.load(f)

            ttl_raw = np.array(data['style_ttl']['data'], dtype=np.float32)
            ttl_dims = data['style_ttl']['dims']
            self.original_ttl = ttl_raw.reshape(ttl_dims).squeeze(0) 

            dp_raw = np.array(data['style_dp']['data'], dtype=np.float32)
            dp_dims = data['style_dp']['dims']
            self.original_dp = dp_raw.reshape(dp_dims) 

            self.lbl_info.setText(os.path.basename(fname))
            self.reset_voice() 

            self.btn_generate.setEnabled(True)
            self.btn_reset.setEnabled(True)
            self.btn_save.setEnabled(True)
            self.status_bar.showMessage(f"Loaded {fname}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to parse JSON:\n{str(e)}")

    def load_voice_library(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, "Select Multiple Voices", VOICES_DIR, "JSON Files (*.json)")
        if not fnames:
            return

        self.voice_library = []
        try:
            for fname in fnames:
                with open(fname, 'r') as f:
                    data = json.load(f)

                ttl_raw = np.array(data['style_ttl']['data'], dtype=np.float32)
                ttl_dims = data['style_ttl']['dims']
                ttl = ttl_raw.reshape(ttl_dims).squeeze(0)

                dp_raw = np.array(data['style_dp']['data'], dtype=np.float32)
                dp_dims = data['style_dp']['dims']
                dp = dp_raw.reshape(dp_dims)

                self.voice_library.append({'ttl': ttl, 'dp': dp, 'name': os.path.basename(fname)})

            count = len(self.voice_library)
            self.lbl_weights.setText("")
            self.lbl_lib_status.setText(f"{count} voices loaded")
            self.lbl_lib_status.setStyleSheet("font-weight: bold;")
            self.btn_remix.setEnabled(count >= 2)
            self.weights_input.setEnabled(count >= 2)
            self.status_bar.showMessage(f"Loaded {count} voices into library.")

        except Exception as e:
            QMessageBox.warning(self, "Library Load Error", str(e))
            self.voice_library = []
            self.lbl_lib_status.setText("Error loading")

    def remix_voices(self):
        if not self.voice_library:
            return

        count = len(self.voice_library)

        # 1. Parse weights input or generate random weights that sum to 1
        weights = self.weights_input.text().strip().split()
        try:
            weights = [float(w) for w in weights]
        except:
            weights = []
        if len(weights) == count and sum(weights) == 1.0:
            weights = np.array(weights)
        else:
            weights = np.random.rand(count)
            weights = weights / np.sum(weights)

        # 2. Initialize accumulators
        # Use the shape of the first voice as reference
        ref_ttl_shape = self.voice_library[0]['ttl'].shape
        ref_dp_shape = self.voice_library[0]['dp'].shape

        mixed_ttl = np.zeros(ref_ttl_shape, dtype=np.float32)
        mixed_dp = np.zeros(ref_dp_shape, dtype=np.float32)

        # 3. Weighted Sum
        txt = ''
        print(f"Remixing {count} voices with weights: {np.round(weights, 2)}")
        for i, voice in enumerate(self.voice_library):
            w = weights[i]
            mixed_ttl += voice['ttl'] * w
            mixed_dp += voice['dp'] * w
            txt += f"{self.voice_library[i]['name'][:-5]}: {w:.2f}  "

        # 4. Update UI
        self.original_ttl = mixed_ttl
        self.original_dp = mixed_dp
        self.heatmap_ttl.set_data(mixed_ttl)

        self.lbl_weights.setText(txt)
        self.lbl_info.setText(f"Remix ({count} voices)")
        self.btn_generate.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.status_bar.showMessage("Voices remixed successfully.")

    def save_voice_json(self):
        if self.heatmap_ttl.get_data() is None: return
        fname, _ = QFileDialog.getSaveFileName(self, "Save Voice Style", VOICES_DIR, "JSON Files (*.json)")
        if not fname: return
        try:
            current_ttl = self.heatmap_ttl.get_data()
            ttl_export = current_ttl[np.newaxis, :, :]
            dp_export = self.original_dp 

            output_data = {
                "style_ttl": {
                    "data": ttl_export.flatten().tolist(),
                    "dims": [1, 50, 256],
                    "type": "float32"
                },
                "style_dp": {
                    "data": dp_export.flatten().tolist(),
                    "dims": [1, 8, 16],
                    "type": "float32"
                },
                "metadata": {
                    "source_file": "LatentExplorer_Modified.wav",
                    "source_sample_rate": 44100,
                    "target_sample_rate": 44100,
                    "extracted_at": datetime.now().isoformat()
                }
            }
            with open(fname, 'w') as f:
                json.dump(output_data, f, indent=2)
            self.status_bar.showMessage(f"Saved voice to {os.path.basename(fname)}")
            QMessageBox.information(self, "Success", "Voice style saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def reset_voice(self):
        if self.original_ttl is not None:
            self.heatmap_ttl.set_data(self.original_ttl.copy())
            self.status_bar.showMessage("Voice reset to original.")

    def start_inference(self):
        text = self.text_input.toPlainText().strip()
        if not text: return
        self.btn_generate.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_bar.showMessage("Generating...")

        current_ttl = self.heatmap_ttl.get_data()[np.newaxis, :, :]
        current_dp = self.original_dp 
        style = Style(current_ttl, current_dp)
        steps = self.spin_steps.value()
        speed = self.sl_speed.value() / 100.0

        self.worker = InferenceThread(self.tts_engine, text, style, steps, speed)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_finished(self, audio_data, sample_rate):
        self.progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.status_bar.showMessage("Playing...")
        try:
            sd.stop()
            sd.play(audio_data, sample_rate)
        except Exception as e:
            self.status_bar.showMessage(f"Playback Error: {e}")

    def on_inference_error(self, err_msg):
        self.progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        QMessageBox.critical(self, "Inference Error", err_msg)
        self.status_bar.showMessage("Error.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LatentExplorerDSP()
    window.show()
    sys.exit(app.exec_())