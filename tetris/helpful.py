import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Cleared Lines')
    plt.plot(scores, label="Cleared Lines")
    plt.plot(mean_scores, label="Last 100 games Mean Cleared Lines")
    plt.plot(losses, label="Model Loss")
    
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(f'{mean_scores[-1]:.3f}'))
    plt.text(len(losses)-1, losses[-1], str(f'{losses[-1]:.6f}'))
    
    
    if max(scores) > 0:
        plt.scatter(len(scores) -scores[::-1].index(max(scores)) -1, max(scores), c="red", s=10)
        plt.text(len(scores) - scores[::-1].index(max(scores)) -1, max(scores), str(f'best: {max(scores)}'))
    # plt.text(len(losses)-1, losses[-1], str(f'{losses[-1]:.3f}'))
    
    plt.show(block=False)
    plt.legend(loc='upper left')
    # plt.savefig("model_64x128x128.png")
    plt.pause(.1)
