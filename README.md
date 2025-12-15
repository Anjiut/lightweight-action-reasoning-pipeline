A Lightweight Multimodal Human Action Reasoning Pipeline for Embodied Agents
This repository presents a lightweight, end-to-end research prototype for human action understanding and semantic reasoning, designed as a foundational component for future embodied agents.

Given short human action videos, the system performs probabilistic pose-based action recognition and subsequently applies LLM-based reasoning to infer human intent, environmental context, and task-level plans in a structured and interpretable manner.

The project emphasizes modularity, uncertainty awareness, and reasoning transparency, explicitly bridging low-level perception with high-level symbolic reasoning.

⸻

📖 Overview

Understanding human actions and intentions is a core challenge in embodied intelligence. A key difficulty lies in reasoning under perception uncertainty while maintaining coherent task-level understanding.

This project explores a modular pipeline that:
	1.	Models perception uncertainty via pose estimation and a probabilistic MLP classifier that outputs action confidence.
	2.	Performs semantic and temporal reasoning using LLM-based agents to infer intent, affordances, and future steps.

The resulting system serves as an interpretable research prototype for studying action reasoning in embodied-agent settings.

⸻

⚙️ Pipeline

Training Stage

Video input
→ frame extraction
→ pose estimation
→ feature vectorization
→ probabilistic MLP training

Inference & Reasoning Stage

New video
→ pose extraction
→ MLP action prediction (with confidence)
→ single-action reasoning (LLM)
→ temporal / task-level reasoning (LLM)
→ structured JSON output

Each module is independently analyzable and replaceable, enabling flexible research extensions.

⸻

🏷️ Supported Action Labels

This prototype uses a small, interpretable action set to prioritize reasoning analysis:
	•	open_door
	•	pick_book
	•	pour_water
	•	walk_stop

These categories are intentionally limited to support rapid experimentation and clear semantic interpretation.

⸻

📂 Project Structure (Core Components)

Perception & Preprocessing
	•	extract_frames.py – extract frames from training videos
	•	extract_new_frames.py – extract frames from test videos
	•	extract_pose.py – OpenPifPaf pose estimation (training)
	•	extract_new_pose.py – OpenPifPaf pose estimation (testing)
	•	build_dataset.py – convert keypoints to fixed-length vectors

Learning
	•	train_mlp.py – train probabilistic MLP classifier
	•	mlp_action_model.pkl – trained model
	•	pose_scaler.pkl – feature scaler

Reasoning & Integration
	•	reasoning_agent.py – LLM-based intent and temporal reasoning
	•	full_pipeline.py – end-to-end pipeline (video → JSON reasoning)

Visualization & Utilities
	•	visualize_frames.py – frame inspection
	•	visualize_pose.py – pose overlay debugging

Note: Video data, extracted frames, and keypoints are not included in the repository.

⸻

💻 Installation

This project was tested with Python 3.10.

Install dependencies using: pip install -r requirements.txt

🔑 OpenAI API Key

The reasoning agent requires an OpenAI API key.

Set it via environment variable: export OPENAI_API_KEY="YOUR_API_KEY"

🚀 How to Run

1. Training Pipeline

Place training videos under videos/, named by action label:

open_door.mp4
pick_book.mp4
pour_water.mp4
walk_stop.mp4

Run the following in order:

python extract_frames.py
python extract_pose.py
python build_dataset.py
python train_mlp.py

This produces the trained classifier and scaler.

⸻

2. Inference & Reasoning Pipeline

Place test videos under new_videos/, then run:

python extract_new_frames.py
python extract_new_pose.py
python full_pipeline.py

The system outputs:
	•	Majority-vote action predictions with confidence
	•	Per-action semantic reasoning (JSON)
	•	Temporal task-level reasoning over action sequences

⸻

📄 Output Format (Example)

The reasoning module produces structured JSON suitable for downstream agent integration:

{
  "action": "pour_water",
  "confidence": 0.92,
  "reasoning": {
    "intent": "To fill the cup with water for drinking",
    "next_step": "Put down the kettle",
    "environment": {
      "scene_type": "Kitchen",
      "key_objects": ["Kettle", "Cup"]
    }
  }
}

⚠️ Limitations & Future Work

Current limitations:
	•	Assumes a single visible human per video
	•	Environment understanding is inferred symbolically rather than visually grounded
	•	Action taxonomy is intentionally small for prototype clarity

Planned extensions:
	•	Bayesian uncertainty modeling and OOD detection
	•	Visual grounding via vision-language models
	•	Multi-agent tracking and interaction reasoning

⸻

📜 License

MIT License.


