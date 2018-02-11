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

### Data collection mode :
 When running the car in training mode, a human driver controls the car driving around the track. This human driving behaviour is captured by the simulator and saved to disk as a time series comprising of the following components at each time step.

The car is equipped with three front facing cameras recording images from the left, centre and right views of the front facing driving view of the car at each time-step.  The driving simulator then saves frames from the three cameras alongside various measurements of the driving behaviour such as **throttle**, **speed** and **steering angle**.


<img src="https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/left.jpg?raw=true" width="250"/> <img src="https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/Centre.jpg?raw=true" width="250"/> <img src="https://github.com/joshwadd/End2end_Driving_Keras/blob/master/Images/right.jpg?raw=true" width="250"/>


Once this data has been collected from the driving simulator, the camera images are used as an input to the deep learning model which attempts to predict the steering angle for the corresponding input in the range [-1, 1].

The tactics for collecting the data from the simulator was to first drive the car around the track in both clockwise and anti clockwise 10 times each, driving optimally in the centre of the road. After this additional laps were recorded by allowing the car to drive off centre into the roadside and then correcting this behaviour. Having such bad driving examples enriches the training data set to allow the model to be able to recover from bad situations in gets into.

The dataset I collected from driving in the simulator contained a total of **23542** samples. Due to the nature of the track, the vast majority of these data samples showed steering angles at/or close to 0.0. This highly bias data set could bias the learning algorithm to perform badly for large comers. To reduce this the data set was balanced by sub-sampling the original set to produce a more balanced distribution of steering angles. This balanced data set contained a total of **7389** samples.

![](https://github.com/joshwadd/End2end_Driving_Keras/blob/master/output_images/steering_distributions.png?raw=true)

<img align ="middle!" src="https://github.com/joshwadd/End2end_Driving_Keras/blob/master/output_images/steering_distributions.png?raw=true" alt="Logo">


<!--stackedit_data:
eyJoaXN0b3J5IjpbMTEzMDAxNzg1MiwtNDAyNTQxNjIyLC0xMD
Q4MDgxNDldfQ==
-->