from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score, roc_curve

from fashionfail.models.prediction_utils import load_tpu_preds
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
    class_names = list(load_categories().values())

    # Create a subplot grid
    num_rows = (len(classes) + 3) // 4
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    fig.suptitle("Histogram of confidence scores - per Class", fontsize=16)
    fig.tight_layout(pad=3.0)

    # Plot histograms for each class
    for i, class_id in enumerate(classes):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        class_data = df_exploded[df_exploded["classes"] == class_id]
        n, bins, patches = ax.hist(class_data["scores"].values, num_bins, density=False)

        ax.set_title(f"{class_id}: {class_names[class_id]:.20} ({class_data.shape[0]})")
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
    ax.set_title("Histogram of confidence scores - overall")

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def plot_confidence_hist_combined():
    pass


def plot_confidence_violin_combined(
    tps1, fps1, tps2, fps2, dataset1_name, dataset2_name
):
    import seaborn as sns

    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Create data and labels for the two datasets
    data = {
        "Data": np.concatenate([tps1, fps1, tps2, fps2]),
        "Dataset": (
            [dataset1_name] * len(tps1)
            + [dataset1_name] * len(fps1)
            + [dataset2_name] * len(tps2)
            + [dataset2_name] * len(fps2)
        ),
        "Type": (
            ["TP"] * len(tps1)
            + ["FP"] * len(fps1)
            + ["TP"] * len(tps2)
            + ["FP"] * len(fps2)
        ),
    }

    # Create a violin plot using Seaborn with split
    sns.violinplot(
        data=data,
        x="Type",
        y="Data",
        hue="Dataset",
        ax=ax,
        inner="quartile",
        split=True,
        scale="count",
    )

    # configure figure axes
    ax.yaxis.grid(True, linestyle="--", linewidth=0.3)
    ax.set_ylabel("confidence")
    ax.set_title("Combined Violin Plot of confidences - overall @IoU=.50")

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
        >>> plot_confidence_violin(tp_scores, fp_scores, show_roc_curve=True)  # ROC curve plot will be shown with overall violin plot.
        >>> plot_confidence_violin(tp_scores, fp_scores)  # Only the overall violin plot will be shown.
    """
    if class_wise:
        _plot_violin_per_class(tp_data, fp_data)
    elif show_roc_curve:
        _plot_violin_overall_with_roc(tp_data, fp_data)
    else:
        _plot_violin_overall(tp_data, fp_data)


def _plot_violin_per_class(tps, fps):
    classes = list(load_categories().values())

    # Create a subplot grid
    num_rows = (len(classes) + 3) // 4
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

    # Plot histograms for each class
    for i, class_id in enumerate(classes):
        row, col = divmod(i, num_cols)
        ax = axs[row, col]

        # Check if both arrays are empty
        if not (tps[i].any() or fps[i].any()):
            ax.axis("off")
            continue
        # Use the actual data if available, otherwise assign a small value (0.0)
        # This ensures the violin plot displays correctly even when data is missing
        tps_data = tps[i] if tps[i].any() else [0.0]
        fps_data = fps[i] if fps[i].any() else [0.0]

        parts = ax.violinplot(
            [tps_data, fps_data], showmedians=False, showextrema=False
        )

        # Set color and transparency of distributions
        colors = ["b", "r"]
        alphas = [0.7, 0.7]
        for p, color, alpha in zip(parts["bodies"], colors, alphas):
            p.set_facecolor(color)
            p.set_alpha(alpha)

        # configure axis
        ax.yaxis.grid(True, linestyle="--", linewidth=0.3)
        ax.set_title(f"{i}: {class_id}")
        ax.set_xticks([1, 2], labels=[f"TP ({len(tps[i])})", f"FP ({len(fps[i])})"])
        ax.set_ylabel("confidence")

    # Hide unused subplots
    for i in range(len(classes), num_rows * num_cols):
        row, col = divmod(i, num_cols)
        fig.delaxes(axs[row, col])

    fig.suptitle("Violin Plot of confidences - per Class @IoU=.50", fontsize=16)
    fig.tight_layout(pad=3.0)
    plt.show()


def _plot_violin_overall(tps, fps):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Plot & configure violin parts
    parts = ax.violinplot([tps, fps], showmedians=False, showextrema=False)
    # Set color and transparency of distributions
    colors = ["b", "r"]
    alphas = [0.7, 0.7]
    for p, color, alpha in zip(parts["bodies"], colors, alphas):
        p.set_facecolor(color)
        p.set_alpha(alpha)

    # configure figure axes
    ax.yaxis.grid(True, linestyle="--", linewidth=0.3)
    ax.set_xticks([1, 2], labels=[f"TP ({len(tps)})", f"FP ({len(fps)})"])
    ax.set_ylabel("confidence")

    # Plot the best threshold as a vertical line
    optimal_threshold, _, _, _, roc_auc = _compute_optimal_threshold(tps, fps)
    ax.axhline(
        y=optimal_threshold,
        color="black",
        linestyle="--",
        label=f"Optimal Threshold: {optimal_threshold:.2f} (AUC = {roc_auc:.2f})",
    )
    ax.legend(fontsize="small")

    ax.set_title("Violin Plot of confidences - overall @IoU=.50")
    plt.show()


def _plot_violin_overall_with_roc(tps, fps):
    (
        optimal_threshold,
        optimal_threshold_index,
        fpr,
        tpr,
        roc_auc,
    ) = _compute_optimal_threshold(tps, fps)

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
    # Set color and transparency of distributions
    colors = ["b", "r"]
    alphas = [0.7, 0.7]
    for p, color, alpha in zip(parts["bodies"], colors, alphas):
        p.set_facecolor(color)
        p.set_alpha(alpha)

    # configure figure axes
    axs[1].yaxis.grid(True, linestyle="--", linewidth=0.3)
    axs[1].set_xticks([1, 2], labels=[f"TP ({len(tps)})", f"FP ({len(fps)})"])
    axs[1].set_ylabel("confidence")
    axs[1].set_title("Violin Plot of confidences - overall @IoU=.50")
    # Plot the best threshold as a vertical line
    axs[1].axhline(
        y=optimal_threshold,
        color="black",
        linestyle="--",
        label=f"Optimal Threshold: {optimal_threshold:.2f} (AUC = {roc_auc:.2f})",
    )
    axs[1].legend(fontsize="small")

    plt.show()


def _compute_optimal_threshold(tps, fps):
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

    return optimal_threshold, optimal_threshold_index, fpr, tpr, roc_auc


def plot_precision_recall_curve(coco_eval, cat_ids: list, iou: float = 0.50):
    """
    Plot Precision-Recall (PR) curves for specified COCO evaluation results.

    This function generates and displays PR curves for multiple categories based on COCO evaluation data.
    The curves illustrate the trade-off between precision and recall at the given IoU threshold.

    Args:
        coco_eval (COCOeval): The COCO evaluation object containing evaluation results.
        cat_ids (list): A list of category IDs for which to plot PR curves.
        iou (float, optional): The IoU (Intersection over Union) threshold. Default is 0.5.

    Returns:
        None

    Example:
        >>> from pycocotools.cocoeval import COCOeval
        >>> cat_ids = [1, 2, 3]  # List of category IDs to plot PR curves for
        >>> coco_eval = COCOeval(...)  # Initialize with your COCO evaluation results
        >>> plot_precision_recall_curve(coco_eval, cat_ids, iou=0.5)
    Example:
        >>> plot_precision_recall_curve(coco_eval, cat_ids=[23], iou=0.5)  # Also works for a single category

    Note:
        - The function uses COCO evaluation data to plot PR curves.
        - It supports plotting PR curves for multiple categories in different colors.
        - Iso-F1 curves are also included in the plot for reference.

    """
    # map iou to idx, e.g. 0.5-->0, 0.95-->9
    iou_idx = _map_iou_to_idx(iou)

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate(f"f1={f_score:0.1f}", xy=(0.9, y[45] + 0.02))

    for catId, color in zip(cat_ids, colors):
        pr = coco_eval.eval["precision"][iou_idx, :, catId, 0, 2]

        pr_display = PrecisionRecallDisplay(
            recall=coco_eval.params.recThrs,
            precision=pr,
            average_precision=np.mean(pr),
        )
        pr_display.plot(ax=ax, name=f"Class: {catId}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = pr_display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title(f"Precision-Recall curve @IoU={iou}")

    plt.show()


def plot_precision_recall_curves(coco_eval):
    """
    Plot Precision-Recall (PR) curves for multiple classes and IoU thresholds.

    This function generates and displays PR curves for multiple object detection classes at two different IoU thresholds
    (0.50 and 0.75). The PR curves illustrate the trade-off between precision and recall.

    Args:
        coco_eval (COCOeval): The COCO evaluation object containing evaluation results.

    Returns:
        None

    Example:
        >>> from pycocotools.cocoeval import COCOeval
        >>> coco_eval = COCOeval(...)  # Initialize with your COCO evaluation results
        >>> plot_precision_recall_curves(coco_eval)

    Note:
        - The function uses COCO evaluation data to plot PR curves for selected classes.
        - PR curves are displayed for both IoU thresholds.
        - The legend indicates the IoU value for each curve.

    """
    # Get classes/categories
    class_ids = list(load_categories().keys())
    class_names = list(load_categories().values())

    # Create a subplot grid
    num_rows = (len(class_ids) + 3) // 4
    num_cols = 4

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    fig.suptitle("Precision-Recall curve per Class - @IoU=[.50, .75]", fontsize=16)
    fig.tight_layout(pad=3.0)

    # Plot histograms for each class
    for i, class_id in enumerate(class_ids):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]

        for iou_idx, iou in zip([0, 5], [".50", ".75"]):
            # Retrieve precision values
            pr = coco_eval.eval["precision"][iou_idx, :, class_id, 0, 2]
            # Skip if no precision calculated
            if len(pr[pr > -1]) == 0:
                ax.axis("off")
                continue

            pr_display = PrecisionRecallDisplay(
                recall=coco_eval.params.recThrs,
                precision=pr,
                average_precision=np.mean(pr),
            )
            pr_display.plot(ax=ax, name=f"IoU={iou}")
            ax.legend(prop={"size": 8}, loc="lower left")
            ax.set_title(f"{class_id}: {class_names[class_id]:.25}")

    # Hide unused subplots
    for i in range(len(class_ids), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row, col])

    plt.show()


def _map_iou_to_idx(iou):
    # Check if the input value is within the specified range
    iou_thresholds = np.arange(0.5, 1, 0.05).round(2)
    if iou not in iou_thresholds:
        raise ValueError(f"IoU value must be one of {iou_thresholds}, got: {iou}.")

    iou_to_idx = {round(iou, 2): idx for idx, iou in enumerate(iou_thresholds)}
    iou_idx = iou_to_idx[iou]

    return iou_idx
