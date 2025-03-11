from generals import GridFactory, PettingZooGenerals
from generals.agents import RandomAgent, ExpanderAgent
from generals.core.action import Action

agent = ExpanderAgent()
npc = RandomAgent()

# Initialize grid factory
grid_factory = GridFactory(
    mode="generalsio",
    min_grid_dims=(15, 15),  # Grid height and width are randomly selected
    max_grid_dims=(23, 23),
    mountain_density=0.2,  # Expected percentage of mountains
    city_density=0.05,  # Expected percentage of cities
    seed=38,  # Seed to generate the same map every time
)

agents = {
    agent.id: agent,
    npc.id: npc,
}

env = PettingZooGenerals(agents=[agent.id, npc.id], grid_factory=grid_factory)

# We can draw custom maps - see symbol explanations in README
grid = """
..#...##..
..A.#..4..
.3...1....
...###....
####...9.B
...###....
.2...5....
....#..6..
..#...##..
"""

# Options are used only for the next game
options = {
    "replay_file": "my_replay",  # Save replay as my_replay.pkl
    # "grid": grid,  # Use the custom map
}

observations, info = env.reset(options=options)
terminated = truncated = False

# For testing, we'll assign different unit types to each agent
agent_unit_types = {agent.id: 0, npc.id: 1}  # cavalry for first agent  # infantry for second agent

# Optional: cycle through unit types
turn_counter = 0

while not (terminated or truncated):
    actions = {}
    for agent_id in env.agents:
        # Ask agent for basic action
        original_action = agents[agent_id].act(observations[agent_id])

        # Since our agents don't know about unit types yet, we'll need to modify their actions
        if not original_action.is_pass():
            # Extract components from the original action
            pass_turn = original_action[0]
            row = original_action[1]
            col = original_action[2]
            direction = original_action[3]
            split = original_action[4] if len(original_action) > 4 else 0

            # Determine unit type for this action
            # Option 1: Fixed unit type per agent
            unit_type_idx = agent_unit_types[agent_id]

            # Option 2: Cycle through unit types (uncomment to use)
            # unit_type_idx = (turn_counter + (0 if agent_id == agent.id else 2)) % 4

            # Create new action with unit type specified
            actions[agent_id] = Action(
                to_pass=pass_turn, row=row, col=col, direction=direction, unit_type_idx=unit_type_idx, to_split=split
            )
        else:
            # Pass actions don't need modification
            actions[agent_id] = original_action

    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render()

    turn_counter += 1

print("Game finished!")
