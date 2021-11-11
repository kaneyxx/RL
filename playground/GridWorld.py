import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=4, help='world size')
parser.add_argument('--discount', type=float, default=1, help='discount rate')
parser.add_argument('--iter', type=int, default=1000, help='iteration count')


# check if episode is done
def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)

# move step
def step(state, action):
    if is_terminal(state):
        return state, 0
    
    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    reward = -1
    return next_state, reward

def draw_multi(async_v, sync_v):
    fig = plt.figure(figsize=(20, 5))
    ax_async = fig.add_subplot(121)
    ax_sync = fig.add_subplot(122)
    ax_async.set_axis_off()
    ax_sync.set_axis_off()
    ax_async.set_title("Asynchronous", pad=30)
    ax_sync.set_title("Synchronous", pad=30)

    tb_async = Table(ax_async, bbox=[0, 0, 1, 1])
    tb_sync = Table(ax_sync, bbox=[0, 0, 1, 1])
    
    nrows, ncols = sync_v.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # async part
    for (i, j), val in np.ndenumerate(async_v):
        tb_async.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # row and column labels
    for i in range(len(async_v)):
        tb_async.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb_async.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax_async.add_table(tb_async)

    # sync part
    for (i, j), val in np.ndenumerate(sync_v):
        tb_sync.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # row and column labels
    for i in range(len(sync_v)):
        tb_sync.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb_sync.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax_sync.add_table(tb_sync)

    plt.savefig('./GridWorld_.png')
    plt.close()


def compute_state_value(in_place=True, discount=1.0, iter_count=1000):
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0
    for _ in range(iter_count):
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-5:
            break

        iteration += 1

    return new_state_values, iteration

def run():
    # async = in-place, sync = out-of-place
    async_values, async_iteration = compute_state_value(in_place=True)
    sync_values, sync_iteration = compute_state_value(in_place=False)

    draw_multi(np.round(sync_values, decimals=0), np.round(async_values, decimals=0))

    print('Asynchronous(in-place): {} iterations'.format(async_iteration))
    print('Synchronous(out-of-place): {} iterations'.format(sync_iteration))



if __name__ == '__main__':
    args = parser.parse_args()

    # size
    WORLD_SIZE = args.size

    # actions for 4 directions
    ACTIONS = [np.array([0, -1]),
            np.array([-1, 0]),
            np.array([0, 1]),
            np.array([1, 0])]
    ACTION_PROB = 0.25

    run()