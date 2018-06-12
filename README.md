# SC2SCOUT
SC2SCOUT learn scout task for the StarCraft II 

## Installation:
pip3 install -e .

## Usage:
cd sc2scout/bin
### train
python3 train_scout.py --map ScoutSimple64 --screen_resolution 64 --agent_race Z --bot_race Z

### evaluate
python3 eval_scout.py --map ScoutSimple64 --screen_resolution 64 --agent_race Z --bot_race Z

### randon agent
python3 zerg_scout_agent.py --map ScoutSimple64 --screen_resolution 64 --agent_race Z --bot_race Z

