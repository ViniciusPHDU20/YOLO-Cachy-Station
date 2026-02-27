# 📟 YOLO STATION X-RAY - NEURAL INTERFACE V12

A **YOLO Station X-Ray** é uma suíte de engenharia de visão computacional de alto desempenho, projetada para automatizar o ciclo de vida completo de modelos YOLOv8. Este software foi arquitetado para fornecer uma interface intuitiva para operadores finais e, ao mesmo tempo, oferecer telemetria de hardware de nível industrial para engenheiros.

---

## 📘 MANUAL DE OPERAÇÃO (PARA USUÁRIOS)

O sistema foi desenvolvido seguindo o protocolo de "Três Ciclos de Ativação". Siga os passos abaixo para iniciar seu processamento:

### 1. Preparação dos Dados (PASSO 1)
*   **O que fazer:** Reúna suas fotos e arquivos de etiquetas (.txt) em uma pasta e compacte-os em um arquivo **.zip**.
*   **Ação:** No aplicativo, clique em `SELECIONAR ARQUIVO ZIP`. O sistema irá descompactar, organizar e validar automaticamente a estrutura necessária para a IA.

### 2. Configuração de Hardware (PASSO 2)
*   **O que fazer:** Definir qual componente do seu computador fará o "trabalho pesado".
*   **Escolha da Placa:**
    *   **NVIDIA CUDA:** Use se você possui uma placa GeForce RTX ou GTX.
    *   **AMD ROCm/DirectML:** Use para placas Radeon.
    *   **CPU Only:** Para computadores sem placa de vídeo dedicada (mais lento).
*   **Ação:** Clique em `CRIAR AMBIENTE DE TRABALHO`. O software baixará os drivers e bibliotecas corretas. *Nota: Isso é feito apenas na primeira vez.*

### 3. Ignição do Motor (PASSO 3)
*   **O que fazer:** Definir a potência do aprendizado.
*   **Potências:** 
    *   **Leve (Nano):** Rápido, ideal para testes ou hardware modesto.
    *   **Equilibrado (Médio):** O padrão para uso comercial.
    *   **Pesado (Grande):** Máxima precisão, exige muita memória de vídeo (VRAM).
*   **Ação:** Clique em `INICIAR TREINAMENTO`. O painel de **Monitoramento** mostrará o progresso em tempo real.

---

## 🛠 ESPECIFICAÇÕES TÉCNICAS E ARQUITETURA

A Estação foi construída utilizando uma pilha tecnológica de elite para garantir portabilidade e isolamento de processos.

### 1. Core Neural & Engine
*   **Framework Base:** Ultralytics YOLOv8 (v8.2.0).
*   **Backend de Tensor:** PyTorch com aceleração via CUDA 12.1 (NVIDIA) ou DirectML/ROCm (AMD).
*   **Patch de Compatibilidade:** Implementamos um *Monkeypatch* exclusivo no `torch.load` para contornar restrições de descompactação do PyTorch 2.6+, garantindo que modelos legados e novos sejam carregados sem interrupções de segurança.

### 2. Interface e UX
*   **Motor Gráfico:** PyQt6 (Qt Framework).
*   **Estilização:** CSS Customizado com tema Dark-Matrix, utilizando fontes mono-espaçadas para leitura técnica clara.
*   **Protocolo de Comunicação:** O aplicativo utiliza o `subprocess.Popen` para gerenciar o motor YOLO em uma thread separada, permitindo que a interface permaneça responsiva durante cargas intensas de processamento.

### 3. Subsistema de Telemetria (Hardware Abstraction Layer)
O monitoramento é feito via captura direta de registros do sistema:
*   **NVIDIA:** Integração nativa com a `pynvml` (NVIDIA Management Library) para leitura de VRAM, Temperatura e Carga de GPU.
*   **AMD (Linux):** Leitura direta dos arquivos de kernel em `/sys/class/drm/card*/device/`, capturando `gpu_busy_percent` e sensores `hwmon` para precisão térmica.
*   **AMD (Windows):** Consulta via `WMIC` (Windows Management Instrumentation) para identificação de adaptador e alocação de memória RAM dedicada.

### 4. Gerenciamento de Ambiente (Sandbox)
*   **Venv Automation:** O software gerencia autonomamente ambientes virtuais Python isolados. Isso impede conflitos entre as versões de bibliotecas da estação e outros programas do seu computador.
*   **Pilha Estável:** Forçamos o uso do **Python 3.11** e **NumPy 1.26.4**, identificada como a versão de maior estabilidade para fluxos de IA em hardware workstation (Xeon/RTX).

---

## 📂 ESTRUTURA DO PROJETO
- `/Linux`: Scripts otimizados para distribuições baseadas em Arch/Debian.
- `/Windows`: Versão nativa com suporte a caminhos de sistema Windows.
- `yolov8_training_runs/`: Repositório automático de resultados, modelos (.pt) e logs de precisão.

*Desenvolvido para operações de visão computacional de alta performance.*
