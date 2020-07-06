# dvd-cover-image-search

Efficient retrieval approach by Nister and Stewenius (pdf of paper provided for reference)

[](https://i.imgur.com/go1Qat2.png =643x375)

Retrieves top 10 matches of an image to DVD covers.

Out of top 10, finds match with most inliers. Plot of image with localized DVD cover.

Download test and training images here: https://drive.google.com/drive/folders/1bZuHDjNAGhhwEFCuxo_GRhGBGc-98KAh?usp=sharing

Homography estimation with RANSAC to match DVD cover image to a test image.

Original:

[](https://i.imgur.com/fZfuNX5.jpg =502x375)

Transform:

[](https://i.imgur.com/UzPVYKs.png =375x502)

Requirements: Python, OpenCV (contrib), Scipy, Numpy
