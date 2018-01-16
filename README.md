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
 When running the car in training mode, a human dr
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwNDY5OTc1NzcsLTEwNDgwODE0OV19
-->