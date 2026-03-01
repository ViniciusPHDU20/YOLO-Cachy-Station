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

# --- BYPASS DE SEGURANÇA PYTORCH 2.6+ (GOD MODE) ---
try:
    import torch
    # Autorizar explicitamente as classes do YOLO para o unpickler do PyTorch
    torch.serialization.add_safe_globals([
        'torch.nn.modules.container.Sequential',
        'torch.nn.modules.conv.Conv2d',
        'torch.nn.modules.batchnorm.BatchNorm2d',
        'torch.nn.modules.activation.SiLU',
        'torch.nn.modules.pooling.MaxPool2d',
        'torch.nn.modules.upsampling.Upsample',
        'torch.nn.modules.container.ModuleList'
    ])
    # Forçar o unpickler a ser permissivo se os globals seguros falharem
    import pickle
    import builtins
    # Redefinir o comportamento de carregamento seguro globalmente
    torch.load = (lambda f, original_load=torch.load, **kwargs: 
                  original_load(f, weights_only=False, **kwargs))
except Exception:
    pass
# --------------------------------------------------

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
QMainWindow { background-color: #000000; }
QWidget { background-color: #000000; color: #00ff41; font-family: 'Fira Code', 'Courier New', monospace; font-size: 13px; }
QGroupBox { border: 1px solid #003b00; border-radius: 5px; margin-top: 20px; font-weight: bold; color: #00ff41; padding: 15px; background-color: #050505; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #00ff41; }
QPushButton { background-color: #001a00; border: 1px solid #008f11; border-radius: 4px; padding: 10px; color: #00ff41; font-weight: bold; }
QPushButton:hover { background-color: #003b00; border: 1px solid #00ff41; color: #00ff41; }
QPushButton#btn_train { background-color: #002b00; border: 2px solid #00ff41; color: #00ff41; text-transform: uppercase; letter-spacing: 1px; }
QLineEdit, QComboBox { background-color: #000000; border: 1px solid #003b00; border-radius: 2px; padding: 8px; color: #00ff41; selection-background-color: #003b00; }
QTextEdit { background-color: #000000; border: 1px solid #001a00; color: #00ff41; font-family: 'Fira Code', monospace; font-size: 12px; }
QProgressBar { border: 1px solid #003b00; border-radius: 2px; text-align: center; background-color: #000; height: 20px; color: #00ff41; }
QProgressBar::chunk { background-color: #00ff41; }
QTabWidget::pane { border: 1px solid #003b00; }
QTabBar::tab { background: #050505; padding: 12px 25px; border: 1px solid #001a00; color: #008f11; }
QTabBar::tab:selected { background: #001a00; border-bottom: 2px solid #00ff41; color: #00ff41; }
QLabel { color: #00ff41; }
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
        self.setWindowTitle(f"YOLO STATION X-RAY - NEURAL INTERFACE ({OS_TYPE})")
        self.setMinimumSize(1100, 850)
        self.setStyleSheet(STYLE_SHEET)
        
        self.workspace = SCRIPT_DIR
        os.chdir(self.workspace)
        
        self.emitter = SignalEmitter()
        self.emitter.output.connect(self.log)
        self.emitter.finished.connect(self.on_process_finished)
        self.emitter.epoch_update.connect(lambda e: self.lbl_epoch.setText(f"EPOC: {e} [IN_PROGRESS]"))
        
        self.train_process = None
        self.nvml_active = False
        self.nvml_handle = None
        
        self.init_ui()
        self.telemetry_timer = QTimer()
        self.telemetry_timer.timeout.connect(self.update_telemetry)
        self.telemetry_timer.start(1000)
        
        self.log(f"SYSTEM_ONLINE // PROTOCOL_V12 // HOST: {platform.node()}")
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
            self.log("✅ NVIDIA_NEURAL_LINK: ESTABLISHED")
        except:
            self.nvml_active = False

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        header = QHBoxLayout()
        header.addWidget(QLabel("<b>DECRYPTING_DATA...</b>"))
        header.addStretch()
        header.addWidget(QLabel(f"ARCH: {platform.machine()} | KERNEL: {platform.version()}"))
        layout.addLayout(header)

        ws_group = QGroupBox("WORKSPACE_IDENTIFIER")
        ws_layout = QHBoxLayout(ws_group)
        self.txt_workspace = QLineEdit(self.workspace)
        self.txt_workspace.setReadOnly(True)
        btn_browse = QPushButton("SCAN_DIR")
        btn_browse.clicked.connect(self.browse_workspace)
        ws_layout.addWidget(self.txt_workspace); ws_layout.addWidget(btn_browse)
        layout.addWidget(ws_group)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.setup_train_tab()
        self.setup_telemetry_tab()
        self.setup_manage_tab()
        
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setMaximumHeight(200)
        layout.addWidget(QLabel("OUTPUT_STREAM:")); layout.addWidget(self.log_area)
        
        footer = QHBoxLayout()
        self.progress_bar = QProgressBar(); footer.addWidget(self.progress_bar)
        self.lbl_status = QLabel("STATUS: IDLE")
        footer.addWidget(self.lbl_status)
        layout.addLayout(footer)

    def setup_train_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        
        # Passo 1: Dataset
        ds_group = QGroupBox("PASSO 1: CARREGAR IMAGENS [DATASET]"); ds_layout = QVBoxLayout(ds_group)
        ds_desc = QLabel("<i>Clique no botão abaixo para importar suas imagens compactadas (.zip).</i>")
        ds_desc.setStyleSheet("font-size: 11px; color: #008f11;")
        btn_zip = QPushButton("📥 SELECIONAR ARQUIVO ZIP")
        btn_zip.setToolTip("Importa o seu conjunto de fotos e etiquetas para a IA aprender.")
        btn_zip.clicked.connect(self.import_dataset)
        ds_layout.addWidget(ds_desc); ds_layout.addWidget(btn_zip); layout.addWidget(ds_group)
        
        # Passo 2: Ambiente
        env_group = QGroupBox("PASSO 2: PREPARAR O COMPUTADOR [AMBIENTE]"); env_layout = QGridLayout(env_group)
        env_desc = QLabel("<i>Escolha sua placa de vídeo e clique em 'CRIAR AMBIENTE'.</i>")
        env_desc.setStyleSheet("font-size: 11px; color: #008f11;")
        self.combo_gpu = QComboBox(); self.combo_gpu.addItems(["NVIDIA CUDA", "AMD DirectML", "CPU Only"])
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["Padrão (Auto)", "Legacy (Xeon)"])
        self.btn_setup = QPushButton("⚡ CRIAR AMBIENTE DE TRABALHO")
        self.btn_setup.setToolTip("Instala as ferramentas necessárias. Faça isso apenas na primeira vez ou se mudar de placa.")
        self.btn_setup.clicked.connect(self.start_setup)
        env_layout.addWidget(env_desc, 0, 0, 1, 2)
        env_layout.addWidget(QLabel("PLACA DE VÍDEO:"), 1, 0); env_layout.addWidget(self.combo_gpu, 1, 1)
        env_layout.addWidget(QLabel("OTIMIZAÇÃO:"), 2, 0); env_layout.addWidget(self.combo_mode, 2, 1)
        env_layout.addWidget(self.btn_setup, 3, 0, 1, 2); layout.addWidget(env_group)
        
        # Passo 3: Treinamento
        tr_group = QGroupBox("PASSO 3: INICIAR APRENDIZADO [TREINO]"); tr_layout = QVBoxLayout(tr_group)
        tr_desc = QLabel("<i>Escolha a potência e clique em 'INICIAR' para a IA começar a estudar as fotos.</i>")
        tr_desc.setStyleSheet("font-size: 11px; color: #008f11;")
        self.combo_profile = QComboBox(); self.combo_profile.addItems(["Leve (Nano)", "Equilibrado (Médio)", "Pesado (Grande)"])
        tr_btns = QHBoxLayout()
        self.btn_train = QPushButton("🚀 INICIAR TREINAMENTO")
        self.btn_train.setObjectName("btn_train")
        self.btn_train.setToolTip("Inicia o motor de inteligência artificial.")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("🛑 PARAR AGORA")
        self.btn_stop.setToolTip("Interrompe o processo imediatamente.")
        self.btn_stop.setStyleSheet("background-color: #330000; border: 1px solid #ff0000;")
        self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self.stop_training)
        tr_btns.addWidget(self.btn_train); tr_btns.addWidget(self.btn_stop)
        self.chk_resume = QPushButton("CONTINUAR DE ONDE PAROU: NÃO"); self.chk_resume.setCheckable(True)
        self.chk_resume.toggled.connect(lambda c: self.chk_resume.setText(f"CONTINUAR DE ONDE PAROU: {'SIM' if c else 'NÃO'}"))
        tr_layout.addWidget(tr_desc)
        tr_layout.addWidget(QLabel("POTÊNCIA DO MODELO:"))
        tr_layout.addWidget(self.combo_profile); tr_layout.addWidget(self.chk_resume); tr_layout.addLayout(tr_btns)
        layout.addWidget(tr_group); self.tabs.addTab(tab, "PAINEL DE CONTROLE")

    def setup_telemetry_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        hw_group = QGroupBox("BIO_SENSOR: HARDWARE_TELEMETRY"); hw_layout = QGridLayout(hw_group)
        
        def s_lbl(t): 
            l = QLabel(t)
            l.setStyleSheet("font-size: 15px; font-weight: bold; color: #00ff41;")
            return l

        self.lbl_gpu_name = s_lbl("GPU: [SCANNING...]"); self.lbl_gpu_vram = s_lbl("VRAM: ---")
        self.lbl_gpu_temp = s_lbl("TEMP: ---"); self.lbl_gpu_load = s_lbl("LOAD: ---")
        self.lbl_cpu_load = s_lbl("CPU_USAGE: ---"); self.lbl_ram_load = s_lbl("RAM_USAGE: ---")
        
        hw_layout.addWidget(self.lbl_gpu_name, 0, 0, 1, 2)
        hw_layout.addWidget(QLabel("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"), 1, 0, 1, 2)
        hw_layout.addWidget(self.lbl_gpu_load, 2, 0); hw_layout.addWidget(self.lbl_gpu_temp, 2, 1)
        hw_layout.addWidget(self.lbl_gpu_vram, 3, 0); hw_layout.addWidget(self.lbl_cpu_load, 3, 1)
        hw_layout.addWidget(self.lbl_ram_load, 4, 0)
        layout.addWidget(hw_group)
        
        det_group = QGroupBox("NEURAL_SYNC: YOLO_REALTIME"); det_layout = QGridLayout(det_group)
        self.lbl_epoch = s_lbl("EPOC: ---"); self.lbl_map = s_lbl("mAP_50: ---")
        det_layout.addWidget(self.lbl_epoch, 0, 0); det_layout.addWidget(self.lbl_map, 0, 1)
        layout.addWidget(det_group); layout.addStretch(); self.tabs.addTab(tab, "BIO_MONITOR")

    def setup_manage_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        btn_nuke = QPushButton("SYSTEM_RESET (ERASE_ALL)"); btn_nuke.setStyleSheet("color: #ff0000; border: 1px solid #ff0000;"); btn_nuke.clicked.connect(self.nuke_workspace)
        layout.addWidget(btn_nuke); layout.addStretch(); self.tabs.addTab(tab, "MAINTENANCE")

    def get_windows_gpu_data(self):
        """Telemetria Genérica via WMIC para Windows (AMD/Intel)"""
        try:
            cmd = "wmic path win32_VideoController get name, AdapterRAM"
            res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            lines = [l.strip() for l in res.stdout.split('\n') if l.strip()]
            if len(lines) > 1:
                parts = lines[1].split()
                vram_bytes = parts[0]
                name = " ".join(parts[1:])
                vram_mb = int(vram_bytes) // (1024**2) if vram_bytes.isdigit() else "-"
                return {"name": name if name else "GENERIC_GPU", "load": "-", "vram": f"{vram_mb} MB", "temp": "-"}
            return {"name": "GENERIC_GPU", "load": "-", "vram": "-", "temp": "-"}
        except: return None

    def update_telemetry(self):
        import psutil
        self.lbl_cpu_load.setText(f"CPU_USAGE: {psutil.cpu_percent()}%")
        self.lbl_ram_load.setText(f"RAM_USAGE: {psutil.virtual_memory().percent}%")
        hw = self.combo_gpu.currentText()
        
        if "NVIDIA" in hw:
            if not self.nvml_active: self.try_init_nvml()
            if self.nvml_active:
                try:
                    import pynvml
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, 0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    name = pynvml.nvmlDeviceGetName(self.nvml_handle)
                    if isinstance(name, bytes): name = name.decode('utf-8')
                    self.lbl_gpu_name.setText(f"GPU: {name}")
                    self.lbl_gpu_vram.setText(f"VRAM: {info.used // (1024**2)} / {info.total // (1024**2)} MB")
                    self.lbl_gpu_temp.setText(f"TEMP: {temp}°C"); self.lbl_gpu_load.setText(f"LOAD: {util.gpu}%")
                except: pass
            else: self.lbl_gpu_name.setText("GPU: NVIDIA [OFFLINE]")
        else:
            win_gpu = self.get_windows_gpu_data()
            if win_gpu:
                self.lbl_gpu_name.setText(f"GPU: {win_gpu['name']}")
                self.lbl_gpu_vram.setText(f"VRAM: {win_gpu['vram']}")
                self.lbl_gpu_temp.setText("-")
                self.lbl_gpu_load.setText("-")

        res_file = os.path.join(self.workspace, RUNS_DIR, "train", "results.csv")
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last = lines[-1].split(",")
                        if not self.train_process:
                            self.lbl_epoch.setText(f"EPOC: {last[0].strip()} [SYNCED]")
                        self.lbl_map.setText(f"mAP_50: {last[6].strip()}")
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
        
        py_code = f"""
import torch
from ultralytics import YOLO
old_load = torch.load
def new_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return old_load(*args, **kwargs)
torch.load = new_load
model = YOLO('{model}')
model.train(data='{yaml_f}', epochs={p['e']}, imgsz={p['i']}, batch={p['b']}, device='{device}', workers=0, amp=False, project='{runs_path}', name='train', exist_ok=True, resume={is_res})
"""
        
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
