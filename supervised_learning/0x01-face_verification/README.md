# Face Verification Project in Keras

## Tasks

### [Utils](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-face_verification/utils.py "Utils")
> Utility package for face verification project.

Contains:
* load_images(images_path, as_array=bool)
        - Loads images from directory according to images_path.
* load_csv(csv_path, params={})
        - List of lists containing contents of csv.
* save_images(path, images, filenames)
        - Saves images with filenames to path parameter.
* generate_triplets(images, filenames, triplet_names)
        - Generates list of triplets based on sublists of triplet_names parameter. Returns [A, P, N]
---

### [Align](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-face_verification/align.py "Align")
> Class built for detecting faces then applying a transformation and rotation for face alignment.

Contains:
* detect(image) - Uses dlib.get_frontal_face_detector to detect face in image.
* find_landmarks(image, detection) - Uses dlib.shape_predictor to find 68 facial landmarks.
* align(image, landmark_indices, anchor_points, size=96) - Aligns an image for face verification.
---

### [TripletLoss](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-face_verification/triplet_loss.py "TripletLoss")
> Class for computing triplet loss for Siamese network. Inherits from keras.layers.Layer.

Contains:
* triplet_loss(inputs) - computes triplet loss for each anchor, positive, and negative in inputs.
---

### [TrainModel](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-face_verification/train_model.py "TrainModel")
> Class that trains Siamese network using triplet loss.

Contains:
* train(triplets, epochs, batch_size) - fit method
* f1_score(y_true, y_pred) - prediction metric
* accuracy(y_true, y_pred) - prediction metric
---

### [FaceVerification](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/0x01-face_verification/verification.py "FaceVerification")
> Class for utilizing trained face verification model.

Contains:
* embedding(images) - Gets logits of image from face verification model.
* verify(image, tau) - returns face verification based on minimum euclidean distance <= tau.



# Examples


### Face Alignment

```python
from align import FaceAlign
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import load_images, save_images

fa = FaceAlign('models/landmarks.dat')
images, filenames = load_images('Image_Directory', as_array=False)
anchors = np.array([[0.194157, 0.16926692], [0.7888591, 0.15817115], [0.4949509, 0.5144414]], dtype=np.float32)
aligned = []
for image in images:
    aligned.append(fa.align(image, np.array([36, 45, 33]), anchors, 96))
aligned = np.array(aligned)
print(aligned.shape)
if not os.path.isdir('New_Aligned_Dir'):
    print(save_images('New_Aligned_Dir', aligned, filenames))
    os.mkdir('New_Aligned_Dir')
print(save_images('New_Aligned_Dir', aligned, filenames))
print(os.listdir('New_Aligned_Dir'))
image = plt.imread('New_Aligned_Dir/KirenSrinivasan.jpg')
plt.imshow(image)
plt.show()
```

### Train Model

```python
import os
import numpy as np
import tensorflow as tf
from train_model import TrainModel
from utils import load_images, load_csv, generate_triplets

images, filenames = load_images('New_Aligned_Dir', as_array=True)
images = images.astype('float32') / 255

triplet_names = load_csv('FVTriplets.csv')
A, P, N = generate_triplets(images, filenames, triplet_names)
triplets = [A[:-2], P[:-2], N[:-2]] 

tm = TrainModel('models/face_verification.h5', 0.2)
tm.train(triplets, epochs=1)
base_model = tm.save('models/trained_fv.h5')
print(base_model is tm.base_model)
print(os.listdir('models'))
```

### Verify Faces
```python
from matplotlib.pyplot import imread, imsave
import numpy as np
import tensorflow as tf
from utils import load_images
from verification import FaceVerification

Dir = "New_Aligned_Dir/"

database_files = [Dir+'HeimerRojas.jpg', Dir+'MariaCoyUlloa.jpg', Dir+'MiaMorton.jpg', Dir+'RodrigoCruz.jpg', Dir+'XimenaCarolinaAndradeVargas.jpg']
database_imgs = np.zeros((5, 96, 96, 3))
for i, f in enumerate(database_files):
    database_imgs[i] = imread(f)

database_imgs = database_imgs.astype('float32') / 255

with tf.keras.utils.CustomObjectScope({'tf': tf}):
    base_model = tf.keras.models.load_model('models/face_verification.h5')
    database_embs = base_model.predict(database_imgs)

test_img_positive = imread(Dir+'HeimerRojas0.jpg').astype('float32') / 255
test_img_negative = imread(Dir+'KirenSrinivasan.jpg').astype('float32') / 255

identities = ['HeimerRojas', 'MariaCoyUlloa', 'MiaMorton', 'RodrigoCruz', 'XimenaCarolinaAndradeVargas']
fv = FaceVerification('models/face_verification.h5', database_embs, identities)
print(fv.verify(test_img_positive, tau=0.5))
print(fv.verify(test_img_negative, tau=0.5))
```

### [Download FVTriplets.csv](https://intranet.hbtn.io/rltoken/K2eKViOeIaz2t56yN5aUiA "Download FVTriplets.csv")
### [Download Images](https://intranet.hbtn.io/rltoken/HUFcrjudLFdlb1njAwE8xw "Download Images")
### [Download face_verification.h5](https://intranet-projects-files.s3.amazonaws.com/holbertonschool-ml/face_verification.h5 "Download face_verification.h5")
### [Download Shape_Predictor](https://intranet.hbtn.io/rltoken/dywCocqd6VbKoVpvUmZXvA "Download Shape_Predictor")