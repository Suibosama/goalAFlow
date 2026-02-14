# GoalFlow - End-to-End Autonomous Driving

## Environment

```bash
# Create conda environment
conda create -n goalflow python=3.10
conda activate goalflow

# Install dependencies
pip install torch torchvision torchaudio
pip install pytorch-lightning
pip install pyyaml
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install tensorboard
pip install nuscenes-devkit
```

## Project Structure

```
goalflow/
├── configs/         # Configuration files
├── data/            # Data directory
├── models/          # Model modules
│   ├── perception/  # TransFuser BEV extraction
│   ├── goal/        # Goal point module
│   └── planning/    # Trajectory planning
├── utils/           # Utility functions
└── scripts/         # Training and evaluation scripts
```

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint path/to/checkpoint
```
