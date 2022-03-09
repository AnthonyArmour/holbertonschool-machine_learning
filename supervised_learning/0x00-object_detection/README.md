[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Object Detection Project
Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Well-researched domains of object detection include face detection and pedestrian detection. Object detection has applications in many areas of computer vision, including image retrieval and video surveillance.


## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| tensorflow         | ^2.6.0  |
| keras              | ^2.6.0  |
| cv2                | ^4.1.0  |

## Model
Add to ./data folder
[yolo](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-ml/yolo.h5)

## Tasks:
In this project I use the yolo v3 algorithm to perform object detection. There are multiple files building on the same class because of the structure of the assignment provided by Holberton school. The entire Yolo class can be found in 7-yolo.py which is linked below. The class methods are documented if you would like to know the inner workings.

### [Yolo Class](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/7-yolo.py "Yolo")

``` python
#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    Yolo = __import__('7-yolo').Yolo

    np.random.seed(0)
    anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[10, 13], [16, 30], [33, 23]]])
    yolo = Yolo('./data/yolo.h5', './data/coco_classes.txt', 0.6, 0.5, anchors)
    predictions, image_paths = yolo.predict('./data/yolo')
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-1.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-2.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-3.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-4.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-5.png)
```
Press the s button to save image:
```
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/images/yolo-6.png)
```
Press the s button to save image:
```
---



