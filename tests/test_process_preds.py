import os

import numpy as np
import pandas as pd
import pytest

from fashionfail.process_preds import (
    _filter_preds_for_classes,
    clean_df_preds,
    load_tpu_preds,
)


# Create a fixture to provide sample predictions data
@pytest.fixture
def sample_preds_data():
    sample_preds = [
        {
            "image_file": "image1.jpg",
            "classes": np.array([1, 2], dtype="int32"),
            "scores": np.array([0.9, 0.8], dtype="int32"),
            "boxes": np.array(
                [[100, 300, 2300, 2000], [200, 1500, 750, 1550]], dtype="float32"
            ),
            "masks": [
                {
                    "size": [2400, 2400],
                    "counts": b"`Ug]38_Z2P1ROh0_O>Aa0_O5L7Hl0cRNVLma1e5`ZN_LUe1m4VYN^Khf1m4hXNYKWg1Q5iWNfKVh1^5",
                },
                {
                    "size": [2400, 2400],
                    "counts": b"\\e[l1_1UY2`0A=YOf0]Ob0QOm0C70132O1O002MM54JN43MO11O1OH9B>A`0J6]Oc0nNS1E;YOoV^a3",
                },
            ],
        },
        {
            "image_file": "image2.jpg",
            "classes": np.array([3], dtype="int32"),
            "scores": np.array([0.7], dtype="int32"),
            "boxes": np.array([60, 820, 250, 150], dtype="float32"),
            "masks": [
                {
                    "size": [2400, 2400],
                    "counts": b"bPWl1a1`X2V1ZOa0@?A>A?B<A?B<G9L3O00I9L5M42NM2M34M00N12O1MJ7N23LK6K431J5G:]Oc0A?G9F;`3",
                }
            ],
        },
    ]
    np.save("test-sample.npy", sample_preds)
    yield "test-sample.npy"
    os.remove("test-sample.npy")


# Test load_tpu_preds function
def test_load_tpu_preds(sample_preds_data):
    df_preds_raw = load_tpu_preds(sample_preds_data)
    assert isinstance(df_preds_raw, pd.DataFrame)
    assert len(df_preds_raw) == 2
    # Add more specific assertions about the DataFrame's columns and contents


# Test _filter_preds_for_classes function
def test_filter_preds_for_classes():
    # Create a sample row with predictions
    sample_row = pd.Series(
        {
            "classes": [1, 27, 34],
            "scores": [0.9, 0.8, 0.7],
            "boxes": [[0, 0, 100, 100], [10, 10, 90, 90], [20, 20, 80, 80]],
            "masks": [
                np.array([[0, 1], [1, 1]]),
                np.array([[1, 0], [0, 1]]),
                np.array([[1, 1], [0, 0]]),
            ],
        }
    )

    (
        filtered_classes,
        filtered_scores,
        filtered_boxes,
        filtered_masks,
    ) = _filter_preds_for_classes(sample_row)
    # Add assertions to check if the filtering was done correctly


# Test clean_df_preds function
def test_clean_df_preds(sample_preds_data):
    # Load sample data
    df_preds_raw = load_tpu_preds(sample_preds_data)

    # Test cleaning and filtering of dataframe
    clean_df_preds(df_preds_raw)
    # Add assertions to check if the cleaning and filtering were done correctly


if __name__ == "__main__":
    pytest.main()
