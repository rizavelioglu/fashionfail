- (2015, AAAI) *Obtaining Well Calibrated Probabilities Using Bayesian Binning* [[1]]
  - introduces **Expected Calibration Error (ECE)** and **Maximum Calibration Error (MCE)**.

- (2018, NIPS workshop) *Relaxed Softmax: Efficient Confidence Auto-Calibration for Safe Pedestrian Detection* [[2]]
  - states that *Expected Calibration Error (ECE)* is not well-suited for object detection.
    > "because there are typically many low-confident detections with a score close to 0, which gives disproportionate
    weight (typically more than 95%) to the first bin, so the resulting ECE value is then mostly given by the error in
    the low-confidence predictions."
  - introduces **Average Calibration Error (ACE)**

- (2020, CVPR workshop) *Multivariate Confidence Calibration for Object Detection* [[3]]
  - introduces **Detection Expected Calibration Error (D-ECE)**

\
Reliability Diagrams (papers to cite, explanation can be taken from [here]):
- Morris H DeGroot and Stephen E Fienberg. The comparison and evaluation of forecasters. The statistician, pages 12–22, 1983.
- Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised learning. In Proceedings of the 22nd international conference on Machine learning, pages 625–632. ACM, 2005
- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning, pages 1321–1330, 2017.


[here]: https://openreview.net/pdf?id=S1lG7aTnqQ#page=4
[1]: https://www.dbmi.pitt.edu/wp-content/uploads/2022/10/Obtaining-well-calibrated-probabilities-using-Bayesian-binning.pdf
[2]: https://openreview.net/pdf?id=S1lG7aTnqQ
[3]: https://arxiv.org/pdf/2004.13546.pdf
