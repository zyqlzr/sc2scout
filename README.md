# SC2SCOUT
SC2SCOUT learn scout task for the StarCraft II 

## Installation:
pip3 install -e .

## Usage:
cd sc2scout/bin
### train
python3 train_scout.py --map ScoutSimple64 --wrapper 'explore_v8' --checkpoint_path './model_v8'  --train_log_dir './log_v8'

### evaluate
python3 eval_scout.py --map ScoutSimple64 --wrapper 'explore_v8' --model_dir './model_save/model'

### randon agent
python3 zerg_scout_agent.py --map ScoutSimple64 --wrapper 'explore_v8'

### train selfplay
python3 train_scout_selfplay.py --map ScoutSimple64Dodge --wrapper explore_v8 --checkpoint_path ./model_v8 --train_log_dir ./log_v8 --norender
