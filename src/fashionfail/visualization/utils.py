import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from fashionfail.process_preds import load_tpu_preds
from fashionfail.utils import load_categories


def plot_confidence_hist(preds_path: str, class_wise: bool = False):
    """
    Plot histograms of softmax probabilities for each class in the predictions.

    Args:
        preds_path (str): Path to the predictions file.
        class_wise (bool, optional): If True, plot confidence histograms per class. If False, plot overall histogram.
            Defaults to False.

    Returns:
        None

    Examples:
        >>> plot_confidence_hist("/path/to/predictions.npy", class_wise=True)

    """
    # Load and preprocess predictions
    df_preds = load_tpu_preds(preds_path, preprocess=True)

    # Select relevant columns and remove samples with no predictions
    df_preds = df_preds[["image_file", "boxes", "scores", "classes"]]
    df_preds = df_preds[df_preds["boxes"].apply(lambda box: box.size != 0)]

    # Explode predictions for each sample and adjust class IDs
    df_exploded = df_preds.explode(["classes", "scores", "boxes"])
    df_exploded["classes"] -= 1

    # Convert columns to appropriate data types
    df_exploded = df_exploded.astype({"classes": "int32", "scores": "float64"})

    # Define the number of bins for the histograms
    num_bins = 20

    if class_wise:
        _plot_hist_per_class(df_exploded, num_bins)
    else:
        _plot_hist_overall(df_exploded, num_bins)


def _plot_hist_per_class(df_exploded, num_bins):
    """
    Plot histograms of softmax probabilities for each class individually.

    Args:
        df_exploded (DataFrame): Exploded DataFrame containing predictions.
        num_bins (int): Number of bins for the histograms.

    Returns:
        None

    """
    # List of unique classes
    classes = sorted(df_exploded.classes.unique())
    class_names = list(load_categories().values())[:28]

    # Create a subplot grid
    num_rows = (len(classes) + 3) // 4
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    fig.suptitle("Histogram of Softmax Probabilities - per Class", fontsize=16)
    fig.tight_layout(pad=3.0)

    # Plot histograms for each class
    for i, class_id in enumerate(classes):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        class_data = df_exploded[df_exploded["classes"] == class_id]
        n, bins, patches = ax.hist(class_data["scores"].values, num_bins, density=False)

        ax.set_title(f"{class_id}: {class_names[class_id]:.25}")
        ax.set_ylabel("Count")

    # Hide unused subplots
    for i in range(len(classes), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row, col])

    plt.show()


def _plot_hist_overall(df_exploded, num_bins):
    """
    Plot an overall histogram of softmax probabilities.

    Args:
        df_exploded (DataFrame): Exploded DataFrame containing predictions.
        num_bins (int): Number of bins for the histogram.

    Returns:
        None

    """
    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(df_exploded.scores.values, num_bins, density=False)

    ax.set_xlabel("Scores (Softmax probabilities)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Softmax Probabilities - overall")

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def plot_confidence_violin(
    tp_data: np.array,
    fp_data: np.array,
    class_wise: bool = False,
    show_roc_curve: bool = False,
):
    """
    Plot violin plots comparing the confidence scores of True Positives (TP) and False Positives (FP).

    Args:
        tp_data (np.array): Array containing confidence scores of True Positives.
        fp_data (np.array): Array containing confidence scores of False Positives.
        class_wise (bool, optional): If True, plot confidence scores per class. If False, plot overall scores.
            Defaults to False.
        show_roc_curve (bool, optional): If True, plot ROC curve plot. This argument has no effect when `class_wise` is True.
            Defaults to False.

    Returns:
        None

    Examples:
        >>> tp_scores = np.array([0.9, 0.8, 0.85, 0.88, 0.92])
        >>> fp_scores = np.array([0.6, 0.75, 0.72, 0.78, 0.65])
        >>> plot_confidence_violin(tp_scores, fp_scores, class_wise=True)  # Only class-wise violin plots will be shown.
        >>> plot_confidence_violin(tp_scores, fp_scores, show_roc_curve=True)  # ROC curve plot will be shown with overall violin plots.
    """
    if class_wise:
        _plot_violin_per_class(tp_data, fp_data)
    elif show_roc_curve:
        _plot_violin_overall_with_roc(tp_data, fp_data)
    else:
        _plot_violin_overall(tp_data, fp_data)


def _plot_violin_per_class(tps, fps):
    # Limit the number of classes to 28 during loading
    classes = list(load_categories().values())[:28]

    # Create a subplot grid
    num_rows = (len(classes) + 3) // 4
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

    # Plot histograms for each class
    for i, class_id in enumerate(classes):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        # Check if the arrays are empty
        if not (tps[i].any() or fps[i].any()):
            ax.axis("off")
            continue
        # TODO: this causes to show wrong value in plots, len(tps[i]) or len(fps[i]). Find a better way
        elif not tps[i].any():
            tps[i] = [0.0]  # Add a small value to make it non-empty
        elif not fps[i].any():
            fps[i] = [0.0]  # Add a small value to make it non-empty

        parts = ax.violinplot([tps[i], fps[i]], showmedians=False, showextrema=False)

        # set color of distributions
        parts["bodies"][0].set_facecolor("b")
        parts["bodies"][0].set_alpha(0.7)
        parts["bodies"][1].set_facecolor("r")
        parts["bodies"][1].set_alpha(0.7)

        # configure axis
        ax.yaxis.grid(True, linestyle="--", linewidth=0.3)
        ax.set_title(f"{i}: {class_id}")
        ax.set_xticks([1, 2], labels=[f"TP ({len(tps[i])})", f"FP ({len(fps[i])})"])
        ax.set_ylabel("confidence")

    # Hide unused subplots
    for i in range(len(classes), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row, col])

    fig.suptitle("Violin Plot of confidences - per Class with IoU@.5", fontsize=16)
    fig.tight_layout(pad=3.0)
    plt.show()


def _plot_violin_overall(tps, fps):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Plot & configure violin parts
    parts = ax.violinplot([tps, fps], showmedians=False, showextrema=False)
    parts["bodies"][0].set_facecolor("b")
    parts["bodies"][0].set_alpha(0.7)
    parts["bodies"][1].set_facecolor("r")
    parts["bodies"][1].set_alpha(0.7)

    # configure figure axes
    ax.yaxis.grid(True, linestyle="--", linewidth=0.3)
    ax.set_xticks([1, 2], labels=[f"TP ({len(tps)})", f"FP ({len(fps)})"])
    ax.set_ylabel("confidence")

    fig.suptitle("Violin Plot of confidences - overall with IoU@.5")
    plt.show()


def _plot_violin_overall_with_roc(tps, fps):
    # Combine TP and FP scores and create labels (1 for TP, 0 for FP)
    scores = np.concatenate([tps, fps])
    labels = np.concatenate([np.ones_like(tps), np.zeros_like(fps)])

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Calculate the area under the ROC curve (AUC)
    roc_auc = roc_auc_score(labels, scores)

    # Find the optimal threshold using Youden's J statistic
    optimal_threshold_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_threshold_index]

    # Create a new figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot the ROC curve
    axs[0].plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    axs[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axs[0].scatter(
        fpr[optimal_threshold_index],
        tpr[optimal_threshold_index],
        c="red",
        marker="o",
        label=f"Optimal Threshold = {optimal_threshold:.2f}",
    )
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].set_title("Receiver Operating Characteristic")
    axs[0].legend(loc="lower right")

    # Plot & configure violin parts
    parts = axs[1].violinplot([tps, fps], showmedians=False, showextrema=False)
    parts["bodies"][0].set_facecolor("b")
    parts["bodies"][0].set_alpha(0.7)
    parts["bodies"][1].set_facecolor("r")
    parts["bodies"][1].set_alpha(0.7)

    # configure figure axes
    axs[1].yaxis.grid(True, linestyle="--", linewidth=0.3)
    axs[1].set_xticks([1, 2], labels=[f"TP ({len(tps)})", f"FP ({len(fps)})"])
    axs[1].set_ylabel("confidence")
    axs[1].set_title("Violin Plot of confidences - overall with IoU@.5")
    # Plot the best threshold as a vertical line
    axs[1].axhline(
        y=optimal_threshold,
        color="black",
        linestyle="--",
        label=f"Optimal Threshold: {optimal_threshold:.2f}",
    )
    axs[1].legend(fontsize="small")

    plt.show()
