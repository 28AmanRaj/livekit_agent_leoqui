# AI Voice Tutor with Whiteboard Canvas

A LiveKit-powered voice AI tutor agent that features a synchronized digital whiteboard canvas and interactive formula simulation widgets.

## Features

- 🎤 **Natural Voice Conversation**: Low-latency voice interaction with interruption handling.
- 🎨 **Digital Whiteboard Canvas**: A structured scene graph system on the backend (`SceneManager`) pushes atomic JSON updates to a lightweight frontend renderer.
- 🎬 **Progressive Teaching Mode**: The tutor reveals concepts and connections node-by-node, keeping speech and visual elements in sync.
- 📊 **Interactive Simulation Widgets**: For quantitative subjects (physics, finance, chemistry, etc.), the tutor can launch interactive charts with real-time slider controls.
- 💾 **Visual Memory**: Snapshots allow the student to navigate back through lessons or compare different diagrams side-by-side.

---

## Architecture

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ LiveKit Client  │◀──b──▶│   Voice Agent   │◀─────▶│  Local Server   │
│ (Web Frontend)  │       │ (tutor_agent.py)│       │  (server.py)    │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              ┌─────▼─────┐                 ┌─────▼─────┐
              │ Deepgram  │                 │  OpenAI   │
              │  STT/TTS  │                 │  GPT-4o   │
              └───────────┘                 └───────────┘
```

---

## Prerequisites

- **Python 3.9** or later (Python 3.13 recommended)
- **LiveKit Cloud Credentials** (or a local LiveKit Server setup)
- **API Keys**:
  - OpenAI API key
  - Deepgram API key

---

## Setup & Local Run

### 1. Synchronize Dependencies
This project uses **uv** for fast package management:
```bash
# Sync all dependencies listed in pyproject.toml
uv sync
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (or update your existing `.env` file):
```bash
# Required
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key

# Optional (for LiveKit Cloud integration)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

### 3. Run the Frontend Token/File Server
The frontend is built using pure HTML5 and Vanilla CSS. To run it locally, start the token server:
```bash
python frontend/server.py
```
This serves the frontend UI and the token endpoint at **`http://localhost:8000`**.

### 4. Run the Agent in Development Mode
Start the python backend agent in development mode:
```bash
uv run python tutor_agent.py dev
```

Open `http://localhost:8000` in your web browser, click **Connect**, and start speaking to your AI Voice Tutor.

---

## Core Whiteboard Capabilities

### Supported Diagram Types & Layouts
- **Flowchart** (`vertical` / `horizontal` / `circular`) - processes, workflows, algorithms.
- **Tree** (`vertical` / `horizontal`) - hierarchies, org charts, binary trees.
- **Mind Map** (`radial`) - concept webs, brainstorming.
- **Timeline** (`timeline`) - historical events, chronologies.
- **State Machine** (`horizontal`) - automata, system states.

### Interactive Widgets
When exploring mathematical or physical concepts, the tutor launches an interactive panel containing graphs like:
- **Simple / Compound Interest**
- **Kinetic Energy**
- **Ideal Gas Law**
- **Ohm's Law**

---

## Deployment to LiveKit Cloud

This project is configured with a `Dockerfile` and `livekit.toml` for easy deployment to production.

### Build and Deploy
If you have the LiveKit CLI installed and authenticated:
```bash
lk agent deploy
```
The deploy process will build the docker image using the optimized `uv` builder, pre-download the Silero VAD models, and publish the worker to your LiveKit Cloud organization.
