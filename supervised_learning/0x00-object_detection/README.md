# Object Detection Project

Using the yolo v3 algorithm to perform object detection.

## Tasks:

### [0. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/0-yolo.py "0. Yolo")
> Create class constructor for Yolo
---

### [1. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/1-yolo.py#:~:text=def%20process_outputs(self%2C%20outputs%2C%20image_size)%3A "1. Yolo")
> process_outputs(self, outputs, image_size): Organize and package output predictions.
---

### [2. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/2-yolo.py#:~:text=filter_boxes(self%2C%20boxes%2C%20box_confidences%2C%20box_class_probs)%3A "2. Yolo")
> filter_boxes(self, boxes, box_confidences, box_class_probs): Filter out top box_confidence score to find box coordinates and classes with highest probabilites.
---

### [3. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/3-yolopy#:~:text=def%20non_max_suppression(self%2C%20filtered_boxes%2C%20box_classes%2C%20box_scores)%3A "2. Yolo")
> non_max_suppression(self, filtered_boxes, box_classes, box_scores): Performs non max suppression on an input of filtered box coordinates, and sorted by class and box score, respectively.
---

### [4. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/4-yolo.py#:~:text=def%20load_images(folder_path)%3A "4. Yolo")
> load_images(folder_path): Loading images
---

### [5. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/5-yolo.py#:~:text=preprocess_images(self%2C%20images)%3A "5. Yolo")
> preprocess_images(self, images): Process images for yolo algorithm
---

### [6. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/6-yolo.py#:~:text=show_boxes(self%2C%20image%2C%20boxes%2C%20box_classes%2C%20box_scores%2C%20file_name)%3A "6. Yolo")
> show_boxes(self, image, boxes, box_classes, box_scores, file_name): Displays the image with all boundary boxes, class names, and box scores.
---

### [7. Yolo](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x00-object_detection/7-yolo.py#:~:text=predict(self%2C%20folder_path)%3A "7. Yolo")
> predict(self, folder_path): Makes detections on images included in folder referenced by <folder_path>.
---



