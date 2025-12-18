import numpy as np
import keras

class DQNAgent():
    def __init__(
            self, 
            env,
            input_size, 
            output_size,
        ):

        self.env = env
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.001

        self.target_network = self._build_model(input_size, output_size)

    

    def load_models(self, target_path):
        self.target_network = keras.models.load_model(target_path)



    def _build_model(self, input_size, output_size):
        model = keras.Sequential()
        model.add(keras.layers.Input((input_size,)))
        model.add(keras.layers.Dense(16, activation = "relu"))
        model.add(keras.layers.Dense(8, activation = "relu"))
        model.add(keras.layers.Dense(output_size, activation = "linear"))
        model.compile(loss = 'mse', optimizer = keras.optimizers.Adam(learning_rate = self.learning_rate))

        return model



    def _flatten_obs(self, obs):
        return np.array(list(obs.values())).reshape(1, self.input_size)
    


    def act(self):
        rewards_breakdown = [[0.0, 0.0, 0.0, 0.0]]
        obs, _ = self.env.reset()
        terminated = False
        truncated = False

        while (not terminated and not truncated):
            action = self.target_network.predict(self._flatten_obs(obs), verbose = 0)[0].argmax()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            obs = next_obs
            rewards_breakdown.append(list(info["rewards"].values()))

        return rewards_breakdown
