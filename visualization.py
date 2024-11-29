from globalvar import *

def plot_histories(history1, history2, labels, start_epoch=5):
    plt.figure(figsize=(12, 6))
    start_index = start_epoch - 1
    epochs1 = range(start_epoch, len(history1.history['loss']) + 1)
    epochs2 = range(start_epoch, len(history2.history['loss']) + 1)

    plt.plot(epochs1, history1.history['loss'][start_index:], label=f'{labels[0]} Training Loss')
    plt.plot(epochs1, history1.history['val_loss'][start_index:], label=f'{labels[0]} Validation Loss')
    plt.plot(epochs2, history2.history['loss'][start_index:], label=f'{labels[1]} Training Loss')
    plt.plot(epochs2, history2.history['val_loss'][start_index:], label=f'{labels[1]} Validation Loss')
    plt.title('Model Losses (Starting from Epoch 5)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
