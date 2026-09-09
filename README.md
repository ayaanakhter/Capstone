# 🕵️‍♂️ Hallucination Monitor

**The Hallucination Monitor** is a Mechanistic Interpretability dashboard designed to catch Large Language Models (LLMs) hallucinating in real-time. 

Instead of treating the AI like a "black box," this project hooks deep into the PyTorch tensor math of the model as it thinks. It monitors three core signals:
1. **Softmax Entropy:** Measures how spread out the model's confidence is across the dictionary.
2. **Gradient Norms:** Calculates mathematical Calculus (Backpropagation) on every word to see if the model's output is highly unstable and sensitive to your input.
3. **MC Dropout Variance:** Scrambles the model's "brain" during generation to see if it changes its story.

It features a high-performance **True Real-Time Streaming** backend and a 3D architectural visualization built with React Three Fiber.

---

## 🚀 How to Replicate This Project Locally

Follow this friendly guide to get the Hallucination Monitor running on your own machine!

### 1. Prerequisites
You will need to have the following installed on your computer:
- **Python 3.10+** (For the AI Backend)
- **Node.js 18+** (For the React Frontend)
- **Git**

### 2. Clone the Repository
Open your terminal and clone this repository:
```bash
git clone https://github.com/ayaanakhter/Capstone.git
cd Capstone
```

### 3. Download the Model Weights (Crucial Step!)
Because AI model weights are gigabytes in size, they cannot be uploaded to GitHub. You must download the Qwen model manually:

1. Go to [Qwen2.5-0.5B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main).
2. Download **every single file** listed on that page.
3. In the root of the `Capstone` folder, create a new folder named exactly: `QWEN`.
4. Place all the files you downloaded into the `QWEN` folder.

Your structure should look like this:
```text
Capstone/
├── QWEN/
│   ├── model.safetensors
│   ├── config.json
│   └── ... (all other files)
├── src/
├── frontend/
└── README.md
```

### 4. Start the AI Backend
The backend runs on PyTorch and FastAPI. 

1. Open a terminal in the root `Capstone` folder.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Activate it (Windows):
   .\venv\Scripts\activate
   # Activate it (Mac/Linux):
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the True Real-Time Streaming server:
   ```bash
   python src/api/main.py
   ```
*(The backend runs on `http://localhost:8000`)*

### 5. Start the 3D Dashboard
The frontend is built with React, Vite, and Three.js.

1. Open a **new** terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Install the Node modules:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

### 6. You're Done! 🎉
Open your browser and navigate to `http://localhost:5173`. You should see the Brutalist Hallucination Monitor dashboard. Type a prompt, and watch the signals extract in real time!

---

**Built by Ayaan Akhter for the Capstone Project.**
