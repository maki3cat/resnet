"""Visualization module for plotting model training histories."""

import matplotlib as plt


def plot_histories_err(history1, history2, labels, start_epoch=1):
    """
    Plot both trainning and testing error rates for two model histories.

    Args:
        history1 (History): Training history of the first model.
        history2 (History): Training history of the second model.
        labels (list): names for the two models.
        start_epoch (int): Starting epoch for the plot (default: 1).
    """
    plt.figure(figsize=(12, 6))
    start_index = start_epoch - 1
    epochs1 = range(start_epoch, len(history1.history['accuracy']) + 1)
    epochs2 = range(start_epoch, len(history2.history['accuracy']) + 1)

    # Calculate error rates
    train_error1 = [1 - acc for acc in history1.history['accuracy'][start_index:]]
    val_error1 = [1 - acc for acc in history1.history['val_accuracy'][start_index:]]
    train_error2 = [1 - acc for acc in history2.history['accuracy'][start_index:]]
    val_error2 = [1 - acc for acc in history2.history['val_accuracy'][start_index:]]

    # Plot with dotted lines for training and solid lines for validation
    # Blue for the first model, red for the second model
    plt.plot(epochs1, train_error1, ':', color='blue', label=f'{labels[0]} Training Error Rate')
    plt.plot(epochs1, val_error1, '-', color='blue', label=f'{labels[0]} Validation Error Rate')
    plt.plot(epochs2, train_error2, ':', color='red', label=f'{labels[1]} Training Error Rate')
    plt.plot(epochs2, val_error2, '-', color='red', label=f'{labels[1]} Validation Error Rate')

    plt.title('Model Error Rates')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_val_err(history1, history2, history3, history4,labels, start_epoch=1):
    """
    Plot validation error rates for four model histories.

    Args:
        history1 (History): Training history of the first model.
        history2 (History): Training history of the second model.
        history3 (History): Training history of the third model.
        history4 (History): Training history of the fourth model.
        labels (list): names for the four models.
        start_epoch (int): Starting epoch for the plot (default: 1).
    """
    plt.figure(figsize=(12, 6))
    start_index = start_epoch - 1
    epochs1 = range(start_epoch, len(history1.history['accuracy']) + 1)
    epochs2 = range(start_epoch, len(history2.history['accuracy']) + 1)
    epochs3 = range(start_epoch, len(history3.history['accuracy']) + 1)
    epochs4 = range(start_epoch, len(history4.history['accuracy']) + 1)


    # Calculate error rates
    val_error1 = [1 - acc for acc in history1.history['val_accuracy'][start_index:]]
    val_error2 = [1 - acc for acc in history2.history['val_accuracy'][start_index:]]
    val_error3 = [1 - acc for acc in history3.history['val_accuracy'][start_index:]]
    val_error4 = [1 - acc for acc in history4.history['val_accuracy'][start_index:]]

    # Plot with dotted lines for training and solid lines for validation
    # Blue for the first model, red for the second model
    plt.plot(epochs1, val_error1, '-', color='green', label=f'{labels[0]} Validation Error Rate')
    plt.plot(epochs2, val_error2, ':', color='green', label=f'{labels[1]} Validation Error Rate')
    plt.plot(epochs3, val_error3, '-', color='purple', label=f'{labels[2]} Validation Error Rate')
    plt.plot(epochs4, val_error4, ':', color='purple', label=f'{labels[3]} Validation Error Rate')

    plt.title('ResNet and Plain-CNN Models Val-Err Rates Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
