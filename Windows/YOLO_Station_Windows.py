#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import venv
import yaml
import zipfile
import threading
import platform
import re
from datetime import datetime

# --- Verificação de Segurança ---
PY_BINARY = "python"
if shutil.which("python3.11"):
    PY_BINARY = "python3.11"

# --- Auto-Instalação de Dependências Básicas (Bootstrap) ---
def bootstrap():
    missing = []
    try: import PyQt6
    except ImportError: missing.append("PyQt6")
    try: import psutil
    except ImportError: missing.append("psutil")
    try: import requests
    except ImportError: missing.append("requests")

    if missing:
        print(f"🚀 Instalando dependências base: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", *missing, "pyyaml", "nvidia-ml-py3"], capture_output=True)

bootstrap()

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QTextEdit, 
                             QProgressBar, QTabWidget, QMessageBox, QGroupBox, QLineEdit, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer

# --- Inteligência de Caminhos ---
OS_TYPE = platform.system()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv_yolov8")

if OS_TYPE == "Windows":
    PYTHON_BIN = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP_BIN = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    YOLO_BIN = os.path.join(VENV_DIR, "Scripts", "yolo.exe")
else:
    PYTHON_BIN = os.path.join(VENV_DIR, "bin", "python")
    PIP_BIN = os.path.join(VENV_DIR, "bin", "pip")
    YOLO_BIN = os.path.join(VENV_DIR, "bin", "yolo")

STYLE_SHEET = """
QMainWindow { background-color: #050505; }
QWidget { background-color: #050505; color: #dcdcdc; font-family: 'Inter', sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #1a1a1a; border-radius: 8px; margin-top: 15px; font-weight: bold; color: #00ff99; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { background-color: #111; border: 1px solid #222; border-radius: 5px; padding: 10px; color: #fff; min-width: 100px; }
QPushButton:hover { background-color: #181818; border: 1px solid #00ff99; }
QPushButton#btn_train { background-color: #002b1b; border: 1px solid #00ff99; font-weight: bold; }
QLineEdit, QComboBox { background-color: #111; border: 1px solid #222; border-radius: 4px; padding: 6px; color: #fff; }
QTextEdit { background-color: #000; border: 1px solid #111; color: #00ff66; font-family: 'Fira Code', monospace; }
QProgressBar { border: 1px solid #1a1a1a; border-radius: 4px; text-align: center; background-color: #111; height: 20px; }
QProgressBar::chunk { background-color: #00ff99; }
QTabWidget::pane { border: 1px solid #1a1a1a; }
QTabBar::tab { background: #111; padding: 12px 25px; border-right: 1px solid #1a1a1a; }
QTabBar::tab:selected { background: #181818; border-bottom: 2px solid #00ff99; }
"""

RUNS_DIR = "yolov8_training_runs"
DATASET_DIR = "dataset"

class SignalEmitter(QObject):
    output = pyqtSignal(str)
    finished = pyqtSignal(int)
    epoch_update = pyqtSignal(str)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"YOLO Station V12 - AI Vision Suite ({OS_TYPE})")
        self.setMinimumSize(1000, 800)
        self.setStyleSheet(STYLE_SHEET)
        
        self.workspace = SCRIPT_DIR
        os.chdir(self.workspace)
        
        self.emitter = SignalEmitter()
        self.emitter.output.connect(self.log)
        self.emitter.finished.connect(self.on_process_finished)
        self.emitter.epoch_update.connect(lambda e: self.lbl_epoch.setText(f"Época: {e} (Processando...)"))
        
        self.train_process = None
        self.nvml_active = False
        self.nvml_handle = None
        
        self.init_ui()
        self.telemetry_timer = QTimer()
        self.telemetry_timer.timeout.connect(self.update_telemetry)
        self.telemetry_timer.start(1000)
        
        self.log(f"Estação Online. SO: {OS_TYPE} | Host: {sys.version.split()[0]}")
        self.try_init_nvml()

    def try_init_nvml(self):
        try:
            try: import pynvml
            except ImportError:
                vlib = os.path.join(VENV_DIR, "Lib", "site-packages")
                if os.path.exists(vlib) and vlib not in sys.path: sys.path.append(vlib)
                import pynvml
            
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_active = True
            self.log("✅ Telemetria NVIDIA Conectada.")
        except:
            self.nvml_active = False

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        ws_group = QGroupBox("Caminho do Projeto")
        ws_layout = QHBoxLayout(ws_group)
        self.txt_workspace = QLineEdit(self.workspace)
        self.txt_workspace.setReadOnly(True)
        btn_browse = QPushButton("Mudar Pasta")
        btn_browse.clicked.connect(self.browse_workspace)
        ws_layout.addWidget(self.txt_workspace); ws_layout.addWidget(btn_browse)
        layout.addWidget(ws_group)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.setup_train_tab()
        self.setup_telemetry_tab()
        self.setup_manage_tab()
        
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setMaximumHeight(150)
        layout.addWidget(QLabel("Terminal:")); layout.addWidget(self.log_area)
        self.progress_bar = QProgressBar(); layout.addWidget(self.progress_bar)

    def setup_train_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        ds_group = QGroupBox("1. Dataset"); ds_layout = QHBoxLayout(ds_group)
        btn_zip = QPushButton("Importar ZIP"); btn_zip.clicked.connect(self.import_dataset)
        ds_layout.addWidget(btn_zip); layout.addWidget(ds_group)
        
        env_group = QGroupBox("2. Ambiente"); env_layout = QVBoxLayout(env_group)
        self.combo_gpu = QComboBox(); self.combo_gpu.addItems(["NVIDIA CUDA", "AMD DirectML", "CPU Only"])
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["Padrão (Auto)", "Legacy (Xeon)"])
        self.btn_setup = QPushButton("PREPARAR AMBIENTE"); self.btn_setup.clicked.connect(self.start_setup)
        env_layout.addWidget(QLabel("Acelerador:")); env_layout.addWidget(self.combo_gpu)
        env_layout.addWidget(QLabel("Otimização:")); env_layout.addWidget(self.combo_mode)
        env_layout.addWidget(self.btn_setup); layout.addWidget(env_group)
        
        tr_group = QGroupBox("3. Treinamento"); tr_layout = QVBoxLayout(tr_group)
        self.combo_profile = QComboBox(); self.combo_profile.addItems(["Leve (Nano)", "Equilibrado (Médio)", "Pesado (Grande)"])
        tr_btns = QHBoxLayout()
        self.btn_train = QPushButton("INICIAR MOTOR"); self.btn_train.setObjectName("btn_train"); self.btn_train.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("PARAR MOTOR"); self.btn_stop.setStyleSheet("background-color: #440000;"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self.stop_training)
        tr_btns.addWidget(self.btn_train); tr_btns.addWidget(self.btn_stop)
        self.chk_resume = QPushButton("Resume: OFF"); self.chk_resume.setCheckable(True); self.chk_resume.toggled.connect(lambda c: self.chk_resume.setText(f"Resume: {'ON' if c else 'OFF'}"))
        tr_layout.addWidget(self.combo_profile); tr_layout.addWidget(self.chk_resume); tr_layout.addLayout(tr_btns)
        layout.addWidget(tr_group); self.tabs.addTab(tab, "Painel de Controle")

    def setup_telemetry_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        hw_group = QGroupBox("Hardware"); hw_layout = QGridLayout(hw_group)
        self.lbl_gpu_name = QLabel("GPU: -"); self.lbl_gpu_vram = QLabel("VRAM: -")
        self.lbl_gpu_temp = QLabel("Temp: -"); self.lbl_gpu_load = QLabel("Carga: -")
        self.lbl_cpu_load = QLabel("CPU: -"); self.lbl_ram_load = QLabel("RAM: -")
        hw_layout.addWidget(self.lbl_gpu_name, 0, 0); hw_layout.addWidget(self.lbl_gpu_load, 0, 1)
        hw_layout.addWidget(self.lbl_gpu_vram, 1, 0); hw_layout.addWidget(self.lbl_gpu_temp, 1, 1)
        hw_layout.addWidget(self.lbl_cpu_load, 2, 0); hw_layout.addWidget(self.lbl_ram_load, 2, 1)
        layout.addWidget(hw_group)
        det_group = QGroupBox("Motor Status"); det_layout = QVBoxLayout(det_group)
        self.lbl_epoch = QLabel("Época: -"); self.lbl_map = QLabel("Precisão: -")
        det_layout.addWidget(self.lbl_epoch); det_layout.addWidget(self.lbl_map)
        layout.addWidget(det_group); layout.addStretch(); self.tabs.addTab(tab, "Monitoramento")

    def setup_manage_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        btn_nuke = QPushButton("LIMPAR TUDO (Reset)"); btn_nuke.setStyleSheet("color: #ff4444;"); btn_nuke.clicked.connect(self.nuke_workspace)
        layout.addWidget(btn_nuke); layout.addStretch(); self.tabs.addTab(tab, "Manutenção")

    def get_windows_gpu_data(self):
        """Telemetria Genérica via WMIC para Windows (AMD/Intel)"""
        try:
            cmd = "wmic path win32_VideoController get name"
            res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            name = res.stdout.split('\n')[1].strip()
            return {"name": name if name else "Generic GPU", "load": "-", "vram": "-", "temp": "-"}
        except: return None

    def update_telemetry(self):
        import psutil
        self.lbl_cpu_load.setText(f"CPU: {psutil.cpu_percent()}%")
        self.lbl_ram_load.setText(f"RAM: {psutil.virtual_memory().percent}%")
        hw = self.combo_gpu.currentText()
        
        if "NVIDIA" in hw:
            if not self.nvml_active: self.try_init_nvml()
            if self.nvml_active:
                try:
                    import pynvml
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, 0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    self.lbl_gpu_name.setText(f"GPU: {pynvml.nvmlDeviceGetName(self.nvml_handle)}")
                    self.lbl_gpu_vram.setText(f"VRAM: {info.used // (1024**2)} / {info.total // (1024**2)} MB")
                    self.lbl_gpu_temp.setText(f"Temp: {temp}°C"); self.lbl_gpu_load.setText(f"Carga: {util.gpu}%")
                except: pass
            else: self.lbl_gpu_name.setText("NVIDIA (Offline)")
        else:
            win_gpu = self.get_windows_gpu_data()
            if win_gpu:
                self.lbl_gpu_name.setText(f"GPU: {win_gpu['name']}")
                self.lbl_gpu_vram.setText("VRAM: Monitorado pelo SO")
                self.lbl_gpu_temp.setText("-")
                self.lbl_gpu_load.setText("-")

        res_file = os.path.join(self.workspace, RUNS_DIR, "train", "results.csv")
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last = lines[-1].split(",")
                        self.lbl_epoch.setText(f"Época: {last[0].strip()} (Finalizada)")
                        self.lbl_map.setText(f"Precisão: {last[6].strip()}")
            except: pass

    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{ts}] {text}")
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "Workspace", self.workspace)
        if path: self.workspace = path; self.txt_workspace.setText(self.workspace)

    def import_dataset(self):
        zip_path, _ = QFileDialog.getOpenFileName(self, "Selecionar ZIP", "", "ZIP (*.zip)")
        if not zip_path: return
        self.log("Extraindo dataset...")
        def task():
            try:
                ds_path = os.path.join(self.workspace, DATASET_DIR)
                if os.path.exists(ds_path): shutil.rmtree(ds_path)
                os.makedirs(ds_path, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for m in z.infolist():
                        if not os.path.realpath(os.path.join(ds_path, m.filename)).startswith(os.path.realpath(ds_path)):
                            raise Exception("ZipSlip!")
                    z.extractall(ds_path)
                items = os.listdir(ds_path)
                if len(items) == 1 and os.path.isdir(os.path.join(ds_path, items[0])):
                    sub = os.path.join(ds_path, items[0])
                    for i in os.listdir(sub): shutil.move(os.path.join(sub, i), ds_path)
                    os.rmdir(sub)
                if not os.path.exists(os.path.join(ds_path, "data.yaml")):
                    for r, d, f in os.walk(ds_path):
                        if "data.yaml" in f: shutil.copy2(os.path.join(r, "data.yaml"), os.path.join(ds_path, "data.yaml")); break
                self.emitter.output.emit("Dataset pronto!"); self.emitter.finished.emit(0)
            except Exception as e: self.emitter.output.emit(f"Erro: {e}"); self.emitter.finished.emit(1)
        threading.Thread(target=task, daemon=True).start()

    def start_setup(self):
        self.log(f"Recriando ambiente...")
        if os.path.exists(VENV_DIR): shutil.rmtree(VENV_DIR)
        subprocess.run([PY_BINARY, "-m", "venv", VENV_DIR])
        mode = self.combo_mode.currentText(); gpu = self.combo_gpu.currentText()
        self.btn_setup.setEnabled(False); self.progress_bar.setRange(0, 0)
        def task():
            try:
                env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
                self.emitter.output.emit("Instalando Pilha Estável (NumPy 1.26.4 + Torch)...")
                if "NVIDIA" in gpu:
                    idx = "https://download.pytorch.org/whl/cu121"
                    if "Legacy" in mode: subprocess.run([PIP_BIN, "install", "numpy==1.26.4", "torch==2.1.2", "torchvision==0.16.2", "--index-url", idx], env=env)
                    else: subprocess.run([PIP_BIN, "install", "numpy==1.26.4", "torch", "torchvision", "--index-url", idx], env=env)
                elif "AMD" in gpu:
                    subprocess.run([PIP_BIN, "install", "numpy==1.26.4", "torch", "torchvision", "onnxruntime-directml"], env=env)
                else: subprocess.run([PIP_BIN, "install", "numpy==1.26.4", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], env=env)
                self.emitter.output.emit("Finalizando dependências..."); subprocess.run([PIP_BIN, "install", "ultralytics==8.2.0", "opencv-python==4.8.1.78", "nvidia-ml-py3", "psutil"], env=env)
                subprocess.run([PIP_BIN, "install", "numpy==1.26.4"], env=env)
                self.emitter.finished.emit(0)
            except Exception as e: self.emitter.output.emit(f"Erro: {e}"); self.emitter.finished.emit(1)
        threading.Thread(target=task, daemon=True).start()

    def stop_training(self):
        if self.train_process: self.train_process.terminate()

    def start_training(self):
        if not os.path.exists(YOLO_BIN): QMessageBox.critical(self, "Erro", "Ambiente não pronto!"); return
        ds_path = os.path.join(self.workspace, DATASET_DIR).replace('\\', '/')
        yaml_f = os.path.join(ds_path, "data.yaml").replace('\\', '/')
        if not os.path.exists(yaml_f): QMessageBox.critical(self, "Erro", "Dataset ausente!"); return
        p = {"Leve (Nano)": {"m": "yolov8n.pt", "e": 50, "i": 416, "b": 16}, "Equilibrado (Médio)": {"m": "yolov8m.pt", "e": 100, "i": 640, "b": 8}, "Pesado (Grande)": {"m": "yolov8l.pt", "e": 150, "i": 640, "b": 4}}[self.combo_profile.currentText()]
        gpu = self.combo_gpu.currentText()
        device = "0" if "NVIDIA" in gpu else "dml" if "AMD" in gpu else "cpu"
        last_w = os.path.join(self.workspace, RUNS_DIR, "train", "weights", "last.pt").replace('\\', '/')
        is_res = self.chk_resume.isChecked() and os.path.exists(last_w)
        model = last_w if (is_res or os.path.exists(last_w)) else p['m']
        runs_path = os.path.join(self.workspace, RUNS_DIR).replace('\\', '/')
        
        py_code = f"import torch; from ultralytics.nn.tasks import DetectionModel; " \
                  f"getattr(torch.serialization, 'add_safe_globals', lambda x: None)([DetectionModel]); " \
                  f"from ultralytics import YOLO; model = YOLO('{model}'); " \
                  f"model.train(data='{yaml_f}', epochs={p['e']}, imgsz={p['i']}, batch={p['b']}, device='{device}', workers=0, amp=False, project='{runs_path}', name='train', exist_ok=True, resume={is_res})"
        
        self.btn_train.setEnabled(False); self.btn_stop.setEnabled(True)
        def task():
            try:
                env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
                self.train_process = subprocess.Popen([PYTHON_BIN, "-c", py_code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in self.train_process.stdout:
                    clean_line = line.strip()
                    self.emitter.output.emit(clean_line)
                    match = re.search(r"(\d+)/(\d+)\s+\d+.\d+G", clean_line)
                    if match: self.emitter.epoch_update.emit(match.group(1))
                self.train_process.wait(); self.emitter.finished.emit(self.train_process.returncode)
            except: self.emitter.finished.emit(1)
        threading.Thread(target=task, daemon=True).start()

    def on_process_finished(self, code):
        self.btn_setup.setEnabled(True); self.btn_train.setEnabled(True); self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(100 if code == 0 else 0)
        self.log(f"Finalizado. (Código {code})")

    def nuke_workspace(self):
        if QMessageBox.question(self, "Nuke", "Reset total?") == QMessageBox.StandardButton.Yes:
            for d in [VENV_DIR, RUNS_DIR, DATASET_DIR]:
                p = os.path.join(self.workspace, d)
                if os.path.exists(p): shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            self.log("Workspace limpo.")

if __name__ == "__main__":
    app = QApplication(sys.argv); window = MainWindow(); window.show(); sys.exit(app.exec())
