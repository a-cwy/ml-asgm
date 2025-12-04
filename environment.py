from typing import Optional
import gymnasium as gym
import numpy as np

"""
ASSUMPTIONS - IMPORTANT

Water heater is rated at 3kW/h
  This translates to a total use of 2700000J of energy per time step(15 minutes)

Water tank is of a rectangular prism with volume of 300L (100cm x 100cm x 30cm)
  Specific heat capacity of water is 4.184J/g/C
  300L = 300kg = 300000g
  1kW per time step increases temperature by 2700000/300000/4.1843 ~ 0.7169658007C

Water tank loses heat over time (ex. U = 2.0)
  Overall heat energy loss in a time step will be q(W) = U(W/m2/K) x A(m2) x deltaT(K)
  Surface area of tank = 13m2
  q = 2 x 13 x (CUR_TEMP - ROOM_TEMP)
  q = 26 x (CUR_TEMP - ROOM_TEMP)

  Consider CUR_TEMP = 65C
  q = 1003.6W

  Change in temperature over a time step,
  -(1003.6 x 15 x 60)/300000/4.184 = -0.72C

  General equation for temperature loss in a time step,
  Loss  = -[(CUR_TEMP - ROOM_TEMP) x HEAT_TRANSFER_COEF x 13 x 15 x 60]/300000/4.184
        = -(CUR_TEMP - ROOM_TEMP) x HEAT_TRANSFER_COEF x 0.009321
"""




class WaterHeaterEnv(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array"
        ],
        "render_fps": 5
    }

    def __init__(self, render_mode = None):
        # Variables & constants
        self.MAX_DAYS = 365
        self.ROOM_TEMP = 26.4
        self.STERILIZATIOJN_TEMP = 70
        self.ELECTRICIY_PER_USE = [0, 0.25, 0.50, 0.75]
        self.HEAT_TRANSFER_COEF = 2.0
        self.USER_TEMP_PREFERENCE = 45.0
        self.MAX_PERMISSIBLE_TEMP = 90.0

        self.USER_SCHEDULE = [
            0, 0, 0, 0,     #24
            0, 0, 0, 0,     #1
            0, 0, 0, 0,     #2
            0, 0, 0, 0,     #3
            0, 0, 0, 0,     #4
            0, 0, 0, 0,     #5
            0, 0, 0, 0,     #6
            0, 0, 0, 0,     #7
            1, 1, 1, 1,     #8
            1, 1, 1, 1,     #9
            1, 1, 0, 0,     #10
            0, 0, 0, 0,     #11
            0, 0, 0, 0,     #12
            0, 0, 1, 1,     #13
            0, 0, 0, 0,     #14
            0, 0, 0, 0,     #15
            0, 0, 0, 0,     #16
            0, 0, 0, 0,     #17
            0, 0, 0, 0,     #18
            0, 0, 0, 0,     #19
            1, 1, 1, 1,     #20
            1, 1, 1, 1,     #21
            0, 0, 0, 0,     #22
            0, 0, 0, 0,     #23
        ]

        self.total_days = 0
        self.day = 0
        self.time = 0
        self.time_since_sterilization = 0
        self.water_tank_temp = 26.4
        self.target_temp = self.USER_TEMP_PREFERENCE
        self.isUsing = False
        self.elementIsActive = False
        self.forecast = False

        self.temp_loss = 0.0
        
        self.reward_vector = [0.0, 0.0, 0.0, 0.0] # Comfort, Hygiene, Energy, Safety

        # Initialize observation space
        self.observation_space = gym.spaces.Dict(
            {
                "day": gym.spaces.Discrete(7),
                "time": gym.spaces.Discrete(96),
                "waterTemperature": gym.spaces.Box(0.0, 100.0, (1,)),
                "targetTemperature": gym.spaces.Box(0.0, 100.0, (1,)),
                "timeSinceSterilization": gym.spaces.Discrete(1),
                "forecast": gym.spaces.Discrete(2)
            }
        )

        # Initialize action space
        self.action_space = gym.spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None



    def _get_obs(self):
        """
        Get the current observation.

        Returns:
        A dictionary containing the current observation.
        """
        return {
            "day": self.day,
            "time": self.time,
            "waterTemperature": np.array(self.water_tank_temp).astype(np.float32),
            "targetTemperature": np.array(self.target_temp).astype(np.float32),
            "timeSinceSterilization": self.time_since_sterilization,
            "forecast": int(self.forecast)
        }



    def _get_info(self):
        """
        Get information from current internal state.

        Returns:
        A dictionary containing additional information.
        """
        return {
            "rewards": {
                "comfort": self.reward_vector[0],
                "hygiene": self.reward_vector[1],
                "energy": self.reward_vector[2],
                "safety": self.reward_vector[3]
            },
            "heat_loss": self.temp_loss
        }
    


    def _calculate_reward(self, action, weights = [5.0, 3.0, 1.0, 1.0]):
        """
        Calculate the rewards for this current timestep
        
        Returns:
        Tuple of each reward type
        """
        comfort = weights[0] * min(self.water_tank_temp - self.target_temp, 1.0) if self.isUsing else 0.0
        hygiene = -weights[1] * min(self.time_since_sterilization - 96.0, 1.0)
        energy = -weights[2] * self.ELECTRICIY_PER_USE[action] 
        safety = -weights[3] * self.water_tank_temp - self.MAX_PERMISSIBLE_TEMP if self.isUsing and self.water_tank_temp > self.MAX_PERMISSIBLE_TEMP else 0.0
        reward_vector = (comfort, hygiene, energy, safety)

        return reward_vector



    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the simulation to start a newpisode.
        Starting temperature for the water tank is randomized to the range (20, 30).

        Returns:
        The initial observation of the environment.
        """
        super().reset(seed=seed)

        self.total_days = 0
        self.day = 0
        self.time = 0
        self.time_since_sterilization = 0
        self.water_tank_temp = (np.random.random() * 10) + 20
        self.target_temp = self.USER_TEMP_PREFERENCE
        self.isUsing = False
        self.elementIsActive = False
        self.forecast = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info



    def step(self, action):
        """
        Increments the simulation by one time step. Action is performed before increment.

        Returns:
        Tuple (observation, reward, terminated, truncated, info)

        observation   - Next observation
        reward        - The reward gained for this timestep
        terminated    - 0: episode is still active
                        1: episode is over
        truncated     - 0: episode is still active
                        1: episode has exceeded max timesteps
        info          - additial information about the current timestep
        """
        assert action in [0, 1, 2, 3], "Action must be discrete integer betwwen 0 and 3 (inclusive)."

        # Perform action
        self.elementIsActive = bool(action)

        # Update isUsing
        self.isUsing = bool(self.USER_SCHEDULE[self.time - 1])

        # Update cycle for day and time
        if self.time == 96:
            self.time = 1

            if self.day == 7:
                self.day = 1
            else:
                self.day += 1

        else:
            self.time += 1

        # Update cycle for water temperature
        self.temp_loss = (self.water_tank_temp - self.ROOM_TEMP) * self.HEAT_TRANSFER_COEF * 0.009321
        self.water_tank_temp -= self.temp_loss
        
        if self.elementIsActive:
            self.water_tank_temp = min(self.water_tank_temp + (0.7169658007 * action), 100.0)

        # Update cycle for overheat time tracker
        if self.water_tank_temp >= self.STERILIZATIOJN_TEMP:
            self.time_since_sterilization = 0
        else:
            self.time_since_sterilization += 1

        # Update cycle for forecast
        self.forecast = bool(self.USER_SCHEDULE[(self.time + 1) % 96])


        # Calculate reward
        self.reward_vector = self._calculate_reward(action)

        observation = self._get_obs()
        reward = sum(self.reward_vector)
        info = self._get_info()

        return observation, reward, False, False, info