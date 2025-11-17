from environment import WaterHeaterEnv
import gymnasium as gym

def init():
    gym.register(
        id = "WaterHeater-v0",
        entry_point = WaterHeaterEnv,
        max_episode_steps = 365 * 96
    )

def format_rewards(reward_breakdown):
    formatted_string = f"""Rewards Breakdown
    Comfort : {reward_breakdown[0]}
    Hygiene : {reward_breakdown[1]}
    Energy  : {reward_breakdown[2]}
    Safety  : {reward_breakdown[3]}
    Total   : {sum(reward_breakdown)}
    """

    return formatted_string