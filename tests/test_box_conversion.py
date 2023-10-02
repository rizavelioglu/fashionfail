import pytest
import torch

from fashionfail.utils import _box_xyxy_to_yxyx, _box_yxyx_to_xyxy, extended_box_convert


def test_custom_box_conversion_functions():
    # Generate some sample bounding boxes in (y1, x1, y2, x2) format
    yxyx_boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0]])

    # Convert to (x1, y1, x2, y2) format
    xyxy_boxes = _box_yxyx_to_xyxy(yxyx_boxes)

    # Expected result in (x1, y1, x2, y2) format
    expected_xyxy = torch.tensor([[20.0, 10.0, 40.0, 30.0], [25.0, 15.0, 45.0, 35.0]])

    # Check if the conversion is correct
    assert torch.all(torch.eq(xyxy_boxes, expected_xyxy))

    # Convert the (x1, y1, x2, y2) format back to (y1, x1, y2, x2) format
    yxyx_boxes_reverted = _box_xyxy_to_yxyx(xyxy_boxes)

    # Check if the reverted conversion matches the original format
    assert torch.all(torch.eq(yxyx_boxes_reverted, yxyx_boxes))

    print("All tests passed!")


def test_extended_box_convert():
    # test yxyx -> xyxy
    yxyx_boxes = torch.tensor([[10.0, 20.0, 20.0, 30.0], [50.0, 60.0, 70.0, 90.0]])
    xyxy_boxes = extended_box_convert(yxyx_boxes, in_fmt="yxyx", out_fmt="xyxy")
    expected_xyxy = torch.tensor([[20.0, 10.0, 30.0, 20.0], [60.0, 50.0, 90.0, 70.0]])
    # Check if the conversion is correct
    assert torch.all(torch.eq(xyxy_boxes, expected_xyxy))

    # test yxyx -> xywh
    xywh_boxes = extended_box_convert(yxyx_boxes, in_fmt="yxyx", out_fmt="xywh")
    expected_xywh = torch.tensor([[20.0, 10.0, 10.0, 10.0], [60.0, 50.0, 30.0, 20.0]])
    # Check if the conversion is correct
    assert torch.all(torch.eq(xywh_boxes, expected_xywh))

    # test xywh -> yxyx
    xywh_boxes_reverted = extended_box_convert(
        xywh_boxes, in_fmt="xywh", out_fmt="yxyx"
    )
    # Check if the conversion is correct
    assert torch.all(torch.eq(yxyx_boxes, xywh_boxes_reverted))

    print("All tests passed!")


if __name__ == "__main__":
    pytest.main()
