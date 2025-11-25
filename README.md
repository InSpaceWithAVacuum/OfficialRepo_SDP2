# LLM-Based Intrusion Detection System

this is a next-generation Intrusion Detection System (IDS) that uses **Large Language Models (ModernBERT)** to analyze network traffic.  
Unlike traditional signature-based IDS, it reads network logs as text, detecting complex attacks like **SQL Injection**, **DDoS**, and **Ransomware** with high precision.
<img width="2538" height="1250" alt="image" src="https://github.com/InSpaceWithAVacuum/OfficialRepo_SDP2/blob/f88b3b8ff54fd575b3af13850fcb0dfb9c5b8b95/img1.jpg" />

---

##  Simulation Note
This project **simulates** network attacks using `replay_batch.py` and the **Edge-IIoT dataset**.  
It does **not sniff live traffic**, ensuring a safe demo environment for research and universities.

---

##  System Architecture
<img width="2015" height="1157" alt="Screenshot 2025-11-24 083055" src="https://github.com/InSpaceWithAVacuum/OfficialRepo_SDP2/blob/f88b3b8ff54fd575b3af13850fcb0dfb9c5b8b95/img2.jpg" />

### 1. The Data Plane (Edge Layer)

This layer is the **frontline defense**, where traffic is generated and enforcement occurs.

- **Traffic Simulator (`replay_batch.py`)**  
  Acts as the network tap. It reads raw logs from the **Edge-IIoT dataset**, injects them into the system, and simulates high-speed traffic bursts (thousands of packets per second).

- **Edge Agent (`edge.agent_api`)**  
  The enforcement point. It receives commands from the Cloud and writes concrete firewall rules (e.g., `BLOCK SRC 192.168.1.5`) to a local `active.rules` file.  
  In real deployments, this can connect directly to `iptables` or a hardware firewall.

---

### 2. The Intelligence Plane (Cloud Layer)

This layer performs **analysis, reasoning, and decision-making**.

- **🧠 Model API (`cloud.model_api`) – The “Brain”**  
  Hosts the fine-tuned **ModernBERT** model, optimized with **ONNX Runtime** for sub-10ms inference.  
  - **Input:** Raw text logs (e.g., “At 10:00, src 10.0.0.1...”)  
  - **Output:** Probabilistic threat classification (e.g., “SQL Injection: 98.4%”), sent to the Policy Engine.

- **⚖️ Policy Engine (`cloud.policy_api`) – The “Judge”**  
  Applies business logic and filtering to model outputs.  
  - **Logic:** Ignores low-confidence detections (e.g., `< 90%`) and maps threats to actions (e.g., *Ransomware → Quarantine*, *Port Scanning → Block IP*).  
  - **Output:** A verified **Action Plan** sent to the Orchestrator.

- **🎼 Orchestrator (`cloud.orchestrator_api`) – The “Commander”**  
  Acts as the **bridge between Cloud and Edge**. It translates Action Plans into concrete enforcement commands that Edge Agents can execute.  
  Ensures accurate, synchronized deployment of defense rules across all edge devices.

### 3. Visualization Plane (SOC)
- ** SOC Dashboard (`dashboard.py`)**: Built with Streamlit, showing live traffic, detections, and defense status.

---

##  Defense Lifecycle

1. **Ingest** → Log generated  
2. **Inference** → Model classifies log  
3. **Verdict** → Policy Engine validates  
4. **Orchestration** → Command sent  
5. **Enforcement** → Edge blocks attacker  
6. **Visualization** → Dashboard updates in real time  

---

## Installation

```bash
git clone https://github.com/YourUsername/sentinel-ids.git
cd sentinel-ids
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Quick Start
(make sure you have .venv activated)
# 1. Microservecies
```bash
uvicorn cloud.model_api:app --port 8000 &
uvicorn cloud.policy_api:app --port 8001 &
uvicorn cloud.orchestrator_api:app --port 8002 &
uvicorn edge.agent_api:app --port 7000 &
```
# 2. Dashboard
```bash
streamlit run dashboard.py
```

# Project Structure
```kotlin
sentinel-ids/
├── dashboard.py
├── requirements.txt
├── data/
│   └── edge_text.parquet
├── cloud/
│   ├── model_api.py
│   ├── policy_api.py
│   └── orchestrator_api.py
├── edge/
│   ├── agent_api.py
│   ├── replay_batch.py
│   └── rules/
└── models/
    └── model.onnx
```
