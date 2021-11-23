import matplotlib.pyplot as plt


def plot(episode, steps, rewards, env):
    plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
    
    plt.plot(episode, steps, '-', color = 'r', label="steps")
    plt.plot(episode, rewards, '-', color = 'g', label="rewards")

    plt.xlim(0, len(episode))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("episode", fontsize=30, labelpad = 15)
    plt.legend(loc = "best", fontsize=20)

    plt.savefig("train_{}.jpg".format(env))