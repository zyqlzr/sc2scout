# PySC2 OpenAI Gym Environments

OpenAI Gym Environments for the StarCraft II PySC2 environment.

## Installation:
pip3 install -e .

## Usage:
train cmd:
cd sc2scout/bin/
python3 train_scout.py --map ScoutSimple64 --screen_resolution 64 --agent_race Z --bot_race Z

evaluate:
cd sc2scout/bin/
python3 eval_scout.py --map ScoutSimple64 --screen_resolution 64 --agent_race Z --bot_race Z

randon agent:
cd sc2scout/bin/
python3 zerg_scout_agent.py --map ScoutSimple64 --screen_resolution 64 --agent_race Z --bot_race Z

## Available environemnts:

