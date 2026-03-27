import json

def get_simulation_config():
    """Reads the simulation configuration from a JSON file."""
    
    with open("sim_config.json", "r") as f:
        sim_config = json.load(f)
    return sim_config