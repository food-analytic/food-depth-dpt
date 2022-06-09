import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def calculate_metrics(prediction, truth, mask, thresholds=[1.05, 1.15, 1.25]):
    accuracies = []
    N = np.sum(mask)
    for threshold in thresholds:
        # Prevent divided by zero warning
        with np.errstate(divide="ignore"):
            # Calculate accuracy
            sum_value = np.sum(
                np.where(
                    mask,
                    (
                        np.where(
                            prediction > truth, prediction / truth, truth / prediction
                        )
                        < threshold
                    ),
                    False,
                )
            )
            absrel = np.mean(
                np.absolute(prediction[mask == 1] - truth[mask == 1]) / truth[mask == 1]
            )
            mae = np.mean(np.absolute(prediction[mask == 1] - truth[mask == 1]))
            accuracy = sum_value / N
            accuracies.append(accuracy)
    return {
        "accuracy": accuracies,
        "absrel": absrel,
        "mae": mae,
    }


def visualize_image(
    image,
    prediction,
    truth,
    figsize=(20, 15),
    fontsize=40,
    norm_value=(1, 4),
    cmap="jet",
):
    norm = colors.Normalize(norm_value[0], norm_value[1])
    diff_norm = colors.Normalize(0.00, 0.25)
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    diff_scalar = cm.ScalarMappable(norm=diff_norm, cmap=cmap)
    scalar.set_array([])

    _, axs = plt.subplots(2, 2, figsize=figsize)

    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Image", fontdict={"fontsize": fontsize})
    axs[0, 0].set_axis_off()

    axs[0, 1].imshow(prediction, cmap=cmap, norm=norm)
    axs[0, 1].set_title("Prediction", fontdict={"fontsize": fontsize})
    axs[0, 1].set_axis_off()
    plt.colorbar(scalar, ax=axs[0, 1])

    axs[1, 0].imshow(truth, cmap=cmap, norm=norm)
    axs[1, 0].set_title("Ground Truth", fontdict={"fontsize": fontsize})
    axs[1, 0].set_axis_off()
    plt.colorbar(scalar, ax=axs[1, 0])

    axs[1, 1].imshow(
        np.where(prediction > truth, prediction / truth, truth / prediction) - 1,
        cmap=cmap,
        norm=diff_norm,
    )
    axs[1, 1].set_title("Error", fontdict={"fontsize": fontsize})
    axs[1, 1].set_axis_off()
    plt.colorbar(diff_scalar, ax=axs[1, 1])

    plt.show()
