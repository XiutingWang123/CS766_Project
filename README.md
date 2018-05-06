# Computer Vision Project

### Model:
YOLO v1, small

### Training

VOC2007 train/val + VOC2012 train/val

### Performance on VOC2007 test set

mAP: 52%

Class Name | Ground Truth | Predicted | True Positives | False Positives | False Negatives | Avg. Precision
--- | --- | --- | --- | --- | --- | ---
aeroplane | 286 | 214 | 199 | 10 | 5 | 0.6908
bicycle | 357 | 238 | 215 | 20 | 3 | 0.5873
bird | 534 | 362 | 310 | 49 | 3 | 0.5514
boat | 349 | 219 | 171 | 43 | 5 | 0.4506
bottle | 496 | 128 |  90 | 30 | 8 | 0.1569
bus | 247 | 168 | 146 | 20 | 2 | 0.5744
car | 1393 | 929 | 830 | 87 | 12 | 0.5853
cat | 370  | 323 | 286 | 37 |  0 | 0.7493
chair | 1221 | 425 | 317 | 98 | 10 | 0.2160
cow | 306 | 204 | 165 | 33 | 6 | 0.4830
diningtable | 263 | 161 | 126 | 34 | 1 | 0.4483
dog | 527 | 425 | 369 | 55 | 1 | 0.6727
horse | 381 | 284 | 268 | 15 | 1 | 0.6952
motorbike | 349 | 228 | 206 | 19 | 3 | 0.5620
person | 4498 | 3347 | 2879 | 383 | 85 | 0.6223
pottedplant | 541 | 202 | 168 | 27 | 7 | 0.2881
sheep | 277 | 172 | 145 | 24 | 3 | 0.5015
sofa | 377 | 142 | 124 | 18 | 0 | 0.3131
train | 304 | 257 | 237 | 20 | 0 | 0.7719
tvmonitor | 352 | 210 | 180 | 29 | 1 | 0.4944
        


### Requirements:
1. Python 3.5
2. OpenCV 3.0
3. TensorFlow 1.5
