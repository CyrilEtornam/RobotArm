
# 🤖 Robot Arm RL Project Guide

Welcome to the **Robot Arm Reinforcement Learning** project!  
This guide will help you set up, understand, and run everything without breaking your laptop (hopefully).  

---

## 📂 Project Structure
```

RobotArm/
│── envs/
│   └── robot\_arm\_env.py      # Custom MuJoCo Gymnasium environment
│
│── lowCostRobotArm/
│   ├── robotScene.xml        # Scene definition (floor, cube, goal)
│   └── low\_cost\_robot\_arm.xml# Robot arm model
│
│── train\_ppo.py              # Script to train PPO agent
│── eval\_policy.py             # Script to evaluate trained agent
│── requirements.txt           # Dependencies
│── README.md                  # This guide
│
├── models/ppo\_robot\_arm/      # Trained models (.zip files)
├── logs/ppo\_robot\_arm/        # Training logs & monitor.csv
└── eval/ppo\_robot\_arm/        # Evaluation results

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone <repo-link>
cd RobotArm
````

If you don’t use GitHub: unzip the shared folder and `cd` into it.

---

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # on Windows
source venv/bin/activate # on Linux/Mac
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

* `gymnasium`
* `mujoco`
* `stable-baselines3`
* `torch`
* `numpy`

---

## 🚀 Running the Project

### 1. Train the Agent

```bash
python train_ppo.py
```

* Model checkpoints will be saved in `models/ppo_robot_arm/`.
* Training logs go into `logs/ppo_robot_arm/monitor.csv`.
* **Note:** training takes hours. Don’t expect miracles immediately.

---

### 2. Evaluate the Trained Agent

```bash
python eval_policy.py
```

* Opens a MuJoCo viewer where you see the arm attempt to grab the cube.
* Prints episode rewards in the terminal.

---

## 📊 Files Explained

* **`robot_arm_env.py`** → Defines how the robot arm interacts with the environment:

  * Action space = joint controls.
  * Observation = gripper + cube + goal + joint states.
  * Reward = closer to cube + closer to goal + bonus for grasp.

* **`train_ppo.py`** → Trains PPO with Stable-Baselines3:

  * Uses `EvalCallback` (saves best model).
  * Uses `CheckpointCallback` (saves periodic snapshots).

* **`eval_policy.py`** → Loads trained model and runs a few test episodes.

* **`.xml files`** → Define robot, cube, and goal positions.

---

## 💾 Sharing Models

Since trained models are large `.zip` files:

* Stored in `models/ppo_robot_arm/`.
* If not in repo, check Google Drive link (to be shared separately).
* To test someone else’s model, place the `.zip` inside this folder and update `MODEL_NAME` in `eval_policy.py`.

---

## 🧩 Common Issues

* **No viewer opens?**
  → Ensure you installed MuJoCo properly and your GPU/CPU supports rendering.

* **Error: Gym unmaintained**
  → We’re already using `gymnasium`, so double-check you didn’t install old `gym`.

* **No movement during eval?**
  → Might be an untrained/random model. Wait for longer training or use a provided trained model.

---

## 📌 Workflow Tips

* **Don’t retrain from scratch unless necessary.** Use provided checkpoints.
* **Use virtual environments** to avoid dependency hell.
* **Check logs** (`logs/ppo_robot_arm/monitor.csv`) for episode reward trends.
