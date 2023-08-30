import matplotlib.pyplot as plt

from fashionfail.process_preds import load_tpu_preds


def plot_hist_softmax_prob(preds_path: str, class_wise: bool = False):
    """
    Plot histograms of softmax probabilities for each class in the predictions.

    :param preds_path: Path to the predictions file
    :param class_wise: Flag to plot histograms per class
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
        plot_hist_per_class(df_exploded, num_bins)
    else:
        plot_overall_hist(df_exploded, num_bins)


def plot_hist_per_class(df_exploded, num_bins):
    # List of unique classes
    classes = sorted(df_exploded.classes.unique())

    # Create a subplot grid
    num_rows = (len(classes) + 3) // 4  # Calculate rows dynamically
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

        ax.set_title(f"Class {class_id}")
        ax.set_xlabel("Scores (Softmax probabilities)")
        ax.set_ylabel("Count")

    # Hide unused subplots
    for i in range(len(classes), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axs[row, col])

    plt.show()


def plot_overall_hist(df_exploded, num_bins):
    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(df_exploded.scores.values, num_bins, density=False)

    ax.set_xlabel("Scores (Softmax probabilities)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Softmax Probabilities - overall")

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
