---


---

# End 2 End Deep Learning for Autonomous Driving in Keras
In this project, I use convolutional deep neural networks to clone driving behavior by training end to end from input camera images to output steering command to an autonomous vehicle. Keras is using for training, validating and testing the model.</p>
The Udacity Self-Driving Car simulator was used for acquiring training data sets of human driving behavior around test tracks. The convolutional neural network was then trained to map input images to steering angles as a regression problem. Once the model has learnt these mappings arising from human behavior it can be used to generate new steering angles online to control the autonomous vehicle in the simulator.
## Project Files

The workflow for building, testing and training the model is composed of following files



| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `DNN_drive_model.py`                    | Builds and trains the CNN with data augmentation and batching of the training data.                  |
| `drive.py`                   | Implements a given CNN to control autonomous car in simulator. Communicates in real time with the simulator receiving current camera image and telemetry data, received data is then used by the CNN to generate model predictions for required control signal to send.                    |
| `model.h5`                 | Model weights saved in Hierarchical Data Format format  containing model architecture Keras understands.             |
| `video.py`                   | Creates a video based on images found in the output directory produced by running `drive.py`                                                                     |

## Data Collection, Augmentation and Preprocessing

The Udacity simulator contains two separate tracks that differ in both difficulty and visual properties of the environment. On both tracks the simulated autonomous car can be operated in either training mode for data collection, or autnomous mode with the CNN generating control signals in real time.

### Data Collection :
 When running the car in training mode, a human driver controls the car driving around the track. This human driving behaviour is captured by the simulator and saved to disk as a time series comprising of the following components at each time step.

The car is equipped with three front facing cameras recording images from the left, centre and right views of the front facing driving view of the car at each time-step.  The driving simulator then saves frames from the three cameras alongside various measurements of the driving behaviour such as **throttle**, **speed** and **steering angle**.


<img src="https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/left.jpg?raw=true" width="250"/> <img src="https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/Centre.jpg?raw=true" width="250"/> <img src="https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/right.jpg?raw=true" width="250"/>


Once this data has been collected from the driving simulator, the camera images are used as an input to the deep learning model which attempts to predict the steering angle for the corresponding input in the range [-1, 1].

The tactics for collecting the data from the simulator was to first drive the car around the track in both clockwise and anti clockwise 10 times each, driving optimally in the centre of the road. After this additional laps were recorded by allowing the car to drive off centre into the roadside and then correcting this behaviour. Having such bad driving examples enriches the training data set to allow the model to be able to recover from bad situations in gets into.

The dataset I collected from driving in the simulator contained a total of **23542** samples. Due to the nature of the track, the vast majority of these data samples showed steering angles at/or close to 0.0. This highly bias data set could bias the learning algorithm to perform badly for large comers. To reduce this the data set was balanced by sub-sampling the original set to produce a more balanced distribution of steering angles. This balanced data set contained a total of **7389** samples.


![](https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/steering_distributions.png?raw=true")


### Data Augmentation :

To prevent the CNN architecture from over-fitting to the training data set and to increase the ability of the model to generalise well to driving encounters it hasn't seen in the training set, a set of data augmentation techniques were used to extend the data-set.


* **Camera Selection and Steering Correction** : As images are recorded from 3 front facing cameras on the front of the car, the scene can be viewed through one of three slighting differing perspectives. One of the three cameras is randomly selected and the steering angle is corrected to match this altered perspective.

``` python
def choose_camera(center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return mpimg.imread(left), steering_angle + steering_correction
    elif choice ==1:
        return mpimg.imread(right), steering_angle - steering_correction

    return mpimg.imread(center), steering_angle
```

* **Horzontial Flip** : Randomly flipping the half images along the center and changing the steering angle rids the training data set of bias due to the circular curvature of the track.
``` python
def flip_image(image, steering_angle):
    if np.random.rand() < 0.5:
        cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle
```

* **Add Shadow** : The conventional network will have to be robust to the presents of shadows in the road. For this purpose I randomly add shadows to the training data in randomised location

``` python
ef add_shadow(image):
    h, w, d = image.shape
    [x1,x2] = np.random.choice(w,2, replace=False)
    y1 = 0
    y2 = h
    xm, ym = np.mgrid[0:h, 0:w]
    mask = np.zeros_like(image[:,:,1])
    mask[(ym - y1)*(x2 - x1) - (y2 - y1)*(xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
```


* **Translate Image** : The image is randomly translated horizontally and vertically and the steering angle is then corrected for this.
``` python
def translate_image(image, steering_angle, x_range, y_range):

    translate_x = x_range * (np.random.rand() - 0.5)
    translate_y = y_range * (np.random.rand() - 0.5)

    steering_angle += translate_x*0.002 #add a small amount of noise
    translation_matrix = np.float32([[1, 0, translate_x],[0, 1, translate_y]])
    height, width, depth = image.shape
    image = cv2.warpAffine(image, translation_matrix, (width, height))
    return image, steering_angle
```


The example results of applying this data augmentation pipeline to the three front facing camera images are shown below.
![](https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/Orginal_2_cameras.png?raw=true)

![](https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/aug1.png?raw=true)

![](https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/aug2.png?raw=true)

![](https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/aug3.png?raw=true)

The data augmentation pipeline was implemented as a generator function in python. Keras allows the use of generator functions to be used heterogeneous manner on the CPU whilst the computing gradients via propagation is performed on the GPU. The generator pipeline can be found in the `data.py` file in the `generator()` function.

### Preprocessing

Each image is cropped before it is fed into the network, removing information that is not useful for steering the car. This is namely the top of the image containing the sky and horizon, and the bottom of the image containing the car itself. The input images were normalized to be in the range of [-1, 1] as to assist the training procedure. All preprocessing was done inside of the DNN architecture in Keras.



## Convolution Neural Network Model 

The Deep learning model used was based around the architecture reported by Nvidia in their [seminal end to end driving paper](https://arxiv.org/pdf/1604.07316.pdf).

The model architecture show very familiar standard structure for convolutional networks. The rational being the convolutional layers in the first half of the network would learn the optimal feature extraction for the images, with the fully connected layers at the end then learning how to control the car. However in practice the disconnection between the convolutional and fully connect






<!--stackedit_data:
eyJoaXN0b3J5IjpbMTIwMDg4MTI0NSwzMzAzNDUyNjgsLTQwMj
U0MTYyMiwtMTA0ODA4MTQ5XX0=
-->