import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Cleared Lines')
    plt.plot(scores, label="Cleared Lines")
    plt.plot(mean_scores, label="Last 100 games Mean Cleared Lines")
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(f'{mean_scores[-1]:.3f}'))
    plt.show(block=False)
    plt.legend(loc='upper left')
    plt.pause(.1)