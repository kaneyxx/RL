import numpy as np
from collections import deque

def run_episode(agent, env, n_episodes, render):
    scores = deque(maxlen=100)
    best_score = 0
    best_step = 0
    best_mean_score = 0
    for episode in range(n_episodes):
        step = 0
        total_rewards = 0
        state = env.reset()
        while True:
            if render:
                env.render()

            # epsilon decay every 20 episodes
            if (episode+1) % 20 == 0:
                agent.decay()

            # store enough memory -> then learn
            while agent.memory_counter <= agent.memory_capacity:
                action = agent.sample_action(state)
                next_state, reward, terminal, _ = env.step(action)
                terminal_mask = 0.0 if terminal else 1.0
                agent.store_transition(state, action, reward/100, next_state, terminal_mask)
                agent.memory_counter += 1
                if terminal:
                    env.reset()
            
            # sample action
            action = agent.sample_action(state)
            next_state, reward, terminal, _ = env.step(action)
            terminal_mask = 0.0 if terminal else 1.0

            # store experience
            agent.store_transition(state, action, reward/100, next_state, terminal_mask)
            agent.memory_counter += 1

            # cumulate reward
            total_rewards += reward

            agent.learn()

            # next state
            state = next_state

            if terminal:
                scores.append(total_rewards)
                mean_score = np.mean(scores)
                if total_rewards >= 190:
                    agent.save()
                if total_rewards > best_score:
                    best_score = total_rewards
                if step+1 > best_step:
                    best_step = step+1
                    # print("Episode {}/{}, best steps {} improved!".format(episode+1, n_episodes, best_step))
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    # print("Episode {}/{}, avg. score {} improved!".format(episode+1, n_episodes, mean_score))
                if (episode+1) % 20 == 0:
                    print('Episode {}/{} finished after {} timesteps, avg. rewards {}, current epsilon {}, best step {}, n_replay {}'
                        .format(episode+1, n_episodes, step+1, mean_score, agent._get_epsilon(), best_step, len(agent.memory)))
                break
            step += 1
        if episode >= 100 and mean_score >= 195:
            agent.save()
            print("Ran {} episodes and solved the problem!".format(episode+1))
            break
        if (episode+1) % n_episodes == 0:
            print("Didn't solve this problem after 1000 episodes :(")
    env.close()


def test_episode(agent, env, render):
    scores = deque(maxlen=100)
    best_score = 0
    best_step = 0
    for i in range(100):
        step = 0
        total_rewards = 0
        state = env.reset()
        while True:
            if render:
                env.render()

            # sample action
            action = agent.predict(state)
            next_state, reward, terminal, info = env.step(action)

            # cumulate reward
            total_rewards += reward

            # next state
            state = next_state

            if terminal:
                scores.append(total_rewards)
                if total_rewards > best_score:
                    best_score = total_rewards
                if step+1 > best_step:
                    best_step = step+1
                # print('Episode {}/{} finished after {} timesteps, total rewards {}, best score {}, best step {}'
                #       .format(i+1, n_episodes, step+1, total_rewards, best_score, best_step))
                break
            step += 1
    mean_score = np.mean(scores)
    print("Average Scores: {}, best score is {} (with {} steps).".format(mean_score, best_score, best_step))
    env.close()