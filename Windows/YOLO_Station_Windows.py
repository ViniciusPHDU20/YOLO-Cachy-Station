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
# --- Verificação de Segurança ---
PY_VERSION = sys.version_info
if not (3, 10) <= (PY_VERSION.major, PY_VERSION.minor) <= (3, 12):
    print(f"⚠️ AVISO: Versão do Python detectada: {PY_VERSION.major}.{PY_VERSION.minor}")
    print("🚀 Versão Recomendada: 3.11")
from datetime import datetime

# --- Auto-Instalação de Dependências Básicas (Bootstrap) ---
def bootstrap():
    try:
        from PyQt6.QtWidgets import QApplication
        import psutil
        import requests
    except ImportError:
        print("🚀 Instalando dependências base para a interface...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyQt6", "psutil", "requests", "pyyaml"], capture_output=True)
        print("✅ Concluído. Reinicie o aplicativo.")
        sys.exit()

bootstrap()

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QTextEdit, 
                             QProgressBar, QTabWidget, QMessageBox, QGroupBox, QLineEdit, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer

# --- Inteligência de Caminhos Multiplataforma ---
OS_TYPE = platform.system() # 'Windows' ou 'Linux'
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

# Tentar carregar telemetria (pynvml)
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

# --- Constantes de Projeto ---
RUNS_DIR = "yolov8_training_runs"
DATASET_DIR = "dataset"
BACKUP_DIR = "model_backups"
DATA_FORCED = "data_forced.yaml"

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

class SignalEmitter(QObject):
    output = pyqtSignal(str)
    finished = pyqtSignal(int)

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
        
        self.train_process = None
        self.has_nvml = HAS_NVML
        
        if self.has_nvml:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.has_nvml = False

        self.init_ui()
        self.telemetry_timer = QTimer()
        self.telemetry_timer.timeout.connect(self.update_telemetry)
        self.telemetry_timer.start(1000)
        self.log(f"Estação Online. SO: {OS_TYPE} | Motor NVML: {'OK' if self.has_nvml else 'OFF'}")

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        ws_group = QGroupBox("Configuração de Workspace")
        ws_layout = QHBoxLayout(ws_group)
        self.txt_workspace = QLineEdit(self.workspace)
        self.txt_workspace.setReadOnly(True)
        btn_browse = QPushButton("Alterar Pasta")
        btn_browse.clicked.connect(self.browse_workspace)
        ws_layout.addWidget(self.txt_workspace)
        ws_layout.addWidget(btn_browse)
        layout.addWidget(ws_group)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.setup_train_tab()
        self.setup_telemetry_tab()
        self.setup_manage_tab()
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        layout.addWidget(QLabel("Terminal de Operações:"))
        layout.addWidget(self.log_area)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

    def setup_train_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 1. Dataset
        ds_group = QGroupBox("1. Gestão de Dataset")
        ds_layout = QHBoxLayout(ds_group)
        btn_zip = QPushButton("Importar Dataset (.zip)")
        btn_zip.clicked.connect(self.import_dataset)
        ds_layout.addWidget(btn_zip)
        layout.addWidget(ds_group)
        
        # 2. Environment
        env_group = QGroupBox("2. Ambiente Virtual")
        env_layout = QVBoxLayout(env_group)
        
        self.combo_gpu = QComboBox()
        self.combo_gpu.addItems(["NVIDIA CUDA", "AMD DirectML", "CPU Only"])
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Padrão (Auto)", "Legacy (Xeon Compat)"])
        
        self.btn_setup = QPushButton("PREPARAR AMBIENTE (Auto-Instalação)")
        self.btn_setup.clicked.connect(self.start_setup)
        
        env_layout.addWidget(QLabel("Acelerador de Hardware:"))
        env_layout.addWidget(self.combo_gpu)
        env_layout.addWidget(QLabel("Modo de Compatibilidade CPU:"))
        env_layout.addWidget(self.combo_mode)
        env_layout.addWidget(self.btn_setup)
        layout.addWidget(env_group)
        
        # 3. Training
        tr_group = QGroupBox("3. Treinamento")
        tr_layout = QVBoxLayout(tr_group)
        self.combo_profile = QComboBox()
        self.combo_profile.addItems(["Leve (Nano)", "Equilibrado (Médio)", "Pesado (Grande)"])
        
        tr_btns_layout = QHBoxLayout()
        self.btn_train = QPushButton("INICIAR MOTOR")
        self.btn_train.setObjectName("btn_train")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("PARAR MOTOR")
        self.btn_stop.setStyleSheet("background-color: #440000; border: 1px solid #ff4444; color: #fff;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        tr_btns_layout.addWidget(self.btn_train)
        tr_btns_layout.addWidget(self.btn_stop)

        self.chk_resume = QPushButton("Modo Resume: OFF")
        self.chk_resume.setCheckable(True)
        self.chk_resume.setStyleSheet("QPushButton:checked { background-color: #004400; border: 1px solid #00ff00; }")
        self.chk_resume.toggled.connect(lambda checked: self.chk_resume.setText(f"Modo Resume: {'ON' if checked else 'OFF'}"))
        
        tr_layout.addWidget(self.combo_profile)
        tr_layout.addWidget(self.chk_resume)
        tr_layout.addLayout(tr_btns_layout)
        layout.addWidget(tr_group)
        
        self.tabs.addTab(tab, "Painel de Controle")

    def setup_telemetry_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        hw_group = QGroupBox("Telemetria em Tempo Real")
        hw_layout = QGridLayout(hw_group)
        self.lbl_gpu_name = QLabel("GPU: -")
        self.lbl_gpu_vram = QLabel("VRAM: -")
        self.lbl_gpu_temp = QLabel("Temp: -")
        self.lbl_gpu_load = QLabel("Carga: -")
        self.lbl_cpu_load = QLabel("CPU: -")
        self.lbl_ram_load = QLabel("RAM: -")
        hw_layout.addWidget(self.lbl_gpu_name, 0, 0); hw_layout.addWidget(self.lbl_gpu_load, 0, 1)
        hw_layout.addWidget(self.lbl_gpu_vram, 1, 0); hw_layout.addWidget(self.lbl_gpu_temp, 1, 1)
        hw_layout.addWidget(self.lbl_cpu_load, 2, 0); hw_layout.addWidget(self.lbl_ram_load, 2, 1)
        layout.addWidget(hw_group)
        
        det_group = QGroupBox("Status do Motor YOLO")
        det_layout = QVBoxLayout(det_group)
        self.lbl_epoch = QLabel("Época: -")
        self.lbl_map = QLabel("Precisão (mAP50): -")
        det_layout.addWidget(self.lbl_epoch); det_layout.addWidget(self.lbl_map)
        layout.addWidget(det_group)
        layout.addStretch()
        self.tabs.addTab(tab, "Telemetria & GPU")

    def setup_manage_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        btn_nuke = QPushButton("Limpar Workspace (Reset Total)")
        btn_nuke.setStyleSheet("color: #ff4444;")
        btn_nuke.clicked.connect(self.nuke_workspace)
        layout.addWidget(btn_nuke)
        layout.addStretch()
        self.tabs.addTab(tab, "Manutenção")

    def update_telemetry(self):
        import psutil
        self.lbl_cpu_load.setText(f"CPU: {psutil.cpu_percent()}%")
        self.lbl_ram_load.setText(f"RAM: {psutil.virtual_memory().percent}%")
        if self.has_nvml:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                self.lbl_gpu_name.setText(f"GPU: {pynvml.nvmlDeviceGetName(self.nvml_handle)}")
                self.lbl_gpu_vram.setText(f"VRAM: {info.used // (1024**2)} / {info.total // (1024**2)} MB")
                self.lbl_gpu_temp.setText(f"Temp: {temp}°C")
                self.lbl_gpu_load.setText(f"Carga: {util.gpu}%")
            except: pass
        
        res_file = os.path.join(self.workspace, RUNS_DIR, "train", "results.csv")
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last = lines[-1].split(",")
                        self.lbl_epoch.setText(f"Época: {last[0].strip()}")
                        self.lbl_map.setText(f"Precisão: {last[6].strip()}")
            except: pass

    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"<span style='color: #555;'>[{ts}]</span> {text}")
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "Selecionar Workspace", self.workspace)
        if path:
            self.workspace = path
            self.txt_workspace.setText(self.workspace)

    def import_dataset(self):
        zip_path, _ = QFileDialog.getOpenFileName(self, "Selecionar ZIP", "", "ZIP Files (*.zip)")
        if not zip_path: return
        self.log(f"Iniciando extração inteligente...")
        def task():
            try:
                ds_path = os.path.join(self.workspace, DATASET_DIR)
                if os.path.exists(ds_path): shutil.rmtree(ds_path)
                os.makedirs(ds_path, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # FIX: Segurança contra ZipSlip
                    for member in z.infolist():
                        target_path = os.path.realpath(os.path.join(ds_path, member.filename))
                        if not target_path.startswith(os.path.realpath(ds_path)):
                            raise Exception(f"Tentativa de Path Traversal detectada no ZIP: {member.filename}")
                    z.extractall(ds_path)
                
                # Normalização
                items = os.listdir(ds_path)
                if len(items) == 1 and os.path.isdir(os.path.join(ds_path, items[0])):
                    sub = os.path.join(ds_path, items[0])
                    for i in os.listdir(sub):
                        src = os.path.join(sub, i)
                        shutil.move(src, ds_path)
                    os.rmdir(sub)
                # Busca data.yaml
                if not os.path.exists(os.path.join(ds_path, "data.yaml")):
                    for r, d, f in os.walk(ds_path):
                        if "data.yaml" in f:
                            shutil.copy2(os.path.join(r, "data.yaml"), os.path.join(ds_path, "data.yaml"))
                            break
                self.emitter.output.emit("Dataset pronto!")
                self.emitter.finished.emit(0)
            except Exception as e:
                self.emitter.output.emit(f"Erro: {e}")
                self.emitter.finished.emit(1)
        threading.Thread(target=task, daemon=True).start()

    def stop_training(self):
        if self.train_process:
            self.log("Parando motor...")
            self.train_process.terminate()

    def start_setup(self):
        if not os.path.exists(VENV_DIR):
            self.log("Criando ambiente virtual isolado...")
            venv.create(VENV_DIR, with_pip=True)
        
        mode = self.combo_mode.currentText()
        gpu = self.combo_gpu.currentText()
        self.log(f"Iniciando auto-configuração ({gpu} | {mode})...")
        self.btn_setup.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        
        def task():
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                # Limpeza
                subprocess.run([PIP_BIN, "uninstall", "-y", "numpy", "ultralytics", "torch", "torchvision", "opencv-python", "onnxruntime-directml"], cwd=self.workspace, env=env)
                
                # Instalação Blindada por Hardware
                if "NVIDIA" in gpu:
                    if "Legacy" in mode:
                        self.emitter.output.emit("Instalando Pilha NVIDIA Legacy (cu121)...")
                        subprocess.run([PIP_BIN, "install", "numpy==1.26.4", "torch==2.1.2", "torchvision==0.16.2", "--index-url", "https://download.pytorch.org/whl/cu121"], env=env)
                    else:
                        self.emitter.output.emit("Instalando Pilha NVIDIA Moderna (CUDA)...")
                        subprocess.run([PIP_BIN, "install", "numpy", "torch", "torchvision"], env=env)
                elif "AMD" in gpu:
                    self.emitter.output.emit("Instalando Pilha AMD DirectML (Windows)...")
                    subprocess.run([PIP_BIN, "install", "numpy==1.26.4", "torch", "torchvision", "onnxruntime-directml"], env=env)
                else: # CPU
                    self.emitter.output.emit("Instalando Pilha CPU Only...")
                    subprocess.run([PIP_BIN, "install", "numpy==1.26.4", "torch", "torchvision"], env=env)
                
                self.emitter.output.emit("Finalizando Ultralytics e Dependências...")
                subprocess.run([PIP_BIN, "install", "ultralytics==8.2.0", "opencv-python==4.8.1.78", "nvidia-ml-py3", "psutil"], env=env)
                subprocess.run([PIP_BIN, "install", "numpy==1.26.4"], env=env) # Re-garantia
                self.emitter.finished.emit(0)
            except Exception as e:
                self.emitter.output.emit(f"Erro: {e}")
                self.emitter.finished.emit(1)
        threading.Thread(target=task, daemon=True).start()

    def start_training(self):
        if not os.path.exists(YOLO_BIN):
            QMessageBox.critical(self, "Erro", "Ambiente não configurado!")
            return
        ds_path = os.path.join(self.workspace, DATASET_DIR)
        orig_yaml = os.path.join(ds_path, "data.yaml")
        if not os.path.exists(orig_yaml):
            QMessageBox.critical(self, "Erro", "Dataset não encontrado!")
            return
        
        # Config de treinamento
        p = {"Leve (Nano)": {"m": "yolov8n.pt", "e": 50, "i": 416, "b": 16},
             "Equilibrado (Médio)": {"m": "yolov8m.pt", "e": 100, "i": 640, "b": 8},
             "Pesado (Grande)": {"m": "yolov8l.pt", "e": 150, "i": 640, "b": 4}}[self.combo_profile.currentText()]
        
        gpu = self.combo_gpu.currentText()
        device = "cpu"
        if "NVIDIA" in gpu:
            device = "0"
        elif "AMD" in gpu:
            device = "dml" # DirectML no Windows
        
        last_w = os.path.join(self.workspace, RUNS_DIR, "train", "weights", "last.pt")
        is_res = self.chk_resume.isChecked() and os.path.exists(last_w)
        model = last_w if (is_res or os.path.exists(last_w)) else p['m']
        
        cmd = [YOLO_BIN, "train", f"model={model}", f"data={orig_yaml}", f"epochs={p['e']}", 
               f"imgsz={p['i']}", f"batch={p['b']}", f"device={device}", "workers=0", "amp=False",
               f"project={os.path.join(self.workspace, RUNS_DIR)}", "name=train", "exist_ok=True"]
        if is_res: cmd.append("resume=True")
        
        self.btn_train.setEnabled(False); self.btn_stop.setEnabled(True)
        def task():
            try:
                env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
                self.train_process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in self.train_process.stdout: self.emitter.output.emit(line.strip())
                self.train_process.wait()
                self.emitter.finished.emit(self.train_process.returncode)
            except: self.emitter.finished.emit(1)
        threading.Thread(target=task, daemon=True).start()

    def on_process_finished(self, code):
        self.btn_setup.setEnabled(True); self.btn_train.setEnabled(True); self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(100 if code == 0 else 0)
        self.log(f"Finalizado (Código {code})")

    def nuke_workspace(self):
        if QMessageBox.question(self, "Nuke", "Limpar tudo?") == QMessageBox.StandardButton.Yes:
            for d in [VENV_DIR, RUNS_DIR, DATASET_DIR, DATA_FORCED]:
                p = os.path.join(self.workspace, d)
                if os.path.exists(p): 
                    if os.path.isdir(p): shutil.rmtree(p)
                    else: os.remove(p)
            self.log("Workspace limpo.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
