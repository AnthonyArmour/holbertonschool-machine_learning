from Connect4_Model.connect4 import ConnectFour
import numpy as np
from Connect4_Model.RL_Model import Policy

class GameTools():

    def __init__(self):
        self.state_memory = None
        self.action_memory = None
        self.inverse_state_memory = None
        self.inverse_action_memory = None
        self.state_history = None
        self.reward_history = None
        self.action_history = None
        self.game = None

    def one_hot_encode(self, Y, classes):
        """
        One hot encode function to be used to reshape
        Y_label vector
        """
        if type(Y) is not np.ndarray or type(classes) is not int:
            return None
        if classes < 2 or classes < np.amax(Y):
            return None
        mat_encode = np.zeros((len(Y), classes))
        for x, label in enumerate(Y):
            mat_encode[x, label] = 1
        return mat_encode

    def get_action(self, policy, state, test=False):
        action, temp_state = None, state

        if self.game.turn == -1:
            temp_state = state * -1

        while action not in self.game.moves():
            probs = policy.predict(temp_state[np.newaxis, ...])[0]
            if test or self.game.turn == -1:
                action = np.argmax(probs)
            action = np.random.choice(7, p=probs)

        return action

    def take_action(self, action):
        done = self.game.make_move(action)
        return self.game.board[..., np.newaxis], done

    def reset(self):
        self.game = ConnectFour()
        self.state_memory = None
        self.action_memory = None
        self.inverse_state_memory = None
        self.inverse_action_memory = None
        return self.game.board[..., np.newaxis], False

    def game_replay_memory(self, state, move):
        # Store game data including inverse game data from oponent
        if self.game.turn == -1:
            if self.state_memory is None:
                self.state_memory = state[np.newaxis, ...]
                self.action_memory = np.array([move])
            else:
                self.state_memory = np.concatenate((self.state_memory, state[np.newaxis, ...]), axis=0)
                self.action_memory = np.concatenate((self.action_memory, np.array([move])), axis=0)
        else:
            if self.inverse_state_memory is None:
                self.inverse_state_memory = state[np.newaxis, ...]*-1
                self.inverse_action_memory = np.array([move])
            else:
                self.inverse_state_memory = np.concatenate((self.inverse_state_memory, state[np.newaxis, ...]*-1), axis=0)
                self.inverse_action_memory = np.concatenate((self.inverse_action_memory, np.array([move])), axis=0)

    def get_discounted_rewards(self):
        gamma = 0.99

        reward_memory = np.ones((self.action_memory.shape[0],))
        inverse_reward_memory = np.ones((self.inverse_action_memory.shape[0],))

        # discount rewards
        # print("positive rewards before", reward_memory)
        reward_memory = reward_memory*(gamma**np.arange(reward_memory.shape[0]))
        # print("positive rewards after gamma", reward_memory)
        inverse_reward_memory = inverse_reward_memory*(gamma**np.arange(inverse_reward_memory.shape[0]))

        # Normalize
        # reward_memory -= reward_memory.mean()
        # inverse_reward_memory -= inverse_reward_memory.mean()

        if self.game.turn == 1:
            reward_memory = -reward_memory
        else:
            inverse_reward_memory = -inverse_reward_memory

        return np.concatenate((reward_memory, inverse_reward_memory), axis=0)

    def append_history(self, states, actions, rewards):
        if self.state_history is None:
            self.state_history = states
            self.action_history = actions
            self.reward_history = rewards
        else:
            self.state_history = np.concatenate((self.state_history, states), axis=0)
            self.action_history = np.concatenate((self.action_history, actions), axis=0)
            self.reward_history = np.concatenate((self.reward_history, rewards), axis=0)

    def train_memory(self, policy, advantage):
        states = np.concatenate((self.state_memory, self.inverse_state_memory), axis=0)
        actions = np.concatenate((self.action_memory, self.inverse_action_memory), axis=0)

        self.append_history(states, actions, advantage)

        if self.reward_history.size < 40:
            return

        rand_query = np.random.choice(np.arange(self.reward_history.size), size=25, replace=False)

        actions = self.one_hot_encode(self.action_history[rand_query, ...], 7)

        policy.fit(
            self.state_history[rand_query, ...], actions,
            sample_weight=self.reward_history[rand_query, ...], epochs=1, verbose=False
            )

        hist_len = self.state_history.shape[0]
        if hist_len > 500:
            cut = hist_len - 500
            self.state_history = self.state_history[cut:, ...]
            self.reward_history = self.reward_history[cut:, ...]
            self.action_history = self.action_history[cut:, ...]

    def run_games(self, iterations=100):
        policy = Policy(7).get_policy_ResConvNet()
        policy.save_weights("./Baseline/Baseline_Network")

        player_1, player_2 = 1, -1

        for i in range(iterations):

            if i % 50 == 0:
                print("Training Games Epoch - ", i)

            state, done = self.reset()

            while done is False:

                action = self.get_action(policy, state)
                next_state, done = self.take_action(action)
                self.game_replay_memory(state, action)
                state = next_state


            rewards = self.get_discounted_rewards()
            # if self.game.turn == -1:
            #     print("win")
            # else:
            #     print("loss")
            # print(rewards)
            self.train_memory(policy, rewards)

            # self.game.show()

        policy.save_weights("./Model/Active_Policy")
        print("Finished training and saving model!\n")


    def test_baseline(self, matches=50):
        wins = 0

        active_policy = Policy(7).get_policy_ResConvNet()
        active_policy.load_weights("./Model/Active_Policy")

        baseline = Policy(7).get_policy_ResConvNet()
        baseline.load_weights("./Baseline/Baseline_Network")

        for i in range(matches):

            if i % 25 == 0:
                print("Baseline Test Epoch - ", i)

            done = False
            self.game = ConnectFour()
            state = self.game.board[..., np.newaxis]

            while done is False:

                if self.game.turn == 1:
                    action = self.get_action(active_policy, state, test=True)
                else:
                    action = self.get_action(baseline, state, test=True)
                
                state, done = self.take_action(action)

            if self.game.turn == -1:
                wins += 1
            

        print("Wins: {} - Losses: {}".format(wins, matches-wins))

    def clear_memory(self):
        del self.state_memory
        del self.action_memory
        del self.inverse_state_memory
        del self.inverse_action_memory
        del self.state_history
        del self.reward_history
        del self.action_history
        del self.game



if __name__ == "__main__":
    Tools = GameTools()
    Tools.run_games(iterations=200)
    Tools.clear_memory()
    Tools.test_baseline(matches=100)

