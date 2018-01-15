---


---

# End 2 End Deep Learning for Autonomous Driving in Keras
In this project, I use convolutional deep neural networks to clone driving behavior by training end to end from input camera images to output steering command to an autonomous vehicle. Keras is using for training, validating and testing the model.</p>
The Udacity Self-Driving Car simulator was used for acquiring training data sets of human driving behavior around test tracks. The convolutional neural network was then trained to map input images to steering angles as a regression problem. Once the model has learnt these mappings arising from human behavior it can be used to generate new steering angles online to control the autonomous vehicle in the simulator.
## Project Files

<table>
<thead>
<tr>
<th>File</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>DNN_drive_model.py</code></td>
<td>Builds and trains the CNN with data augmentation and batching of the training data.</td>
</tr>
<tr>
<td><code>drive.py</code></td>
<td>Runs a given CNN to control autonomous car in simulator.</td>
</tr>
<tr>
<td><code>model.h5</code></td>
<td>Model weights.</td>
</tr>
<tr>
<td><code>video.py</code></td>
<td>Creates a video based on images found in the output directory produced by running <code>drive.py</code></td>
</tr>
</tbody>
</table>



| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `DNN_drive_model.py`                    | Builds and trains the CNN with data augmentation and batching of the training data.                  |
| `drive.py`                   | Implements a given CNN to control autonomous car in simulator. Communicates in real time with the simulator receiving current camera image and telemetry data, received data is then used by the CNN to generate model predictions for required control signal to send.                    |
| `model.h5`                 | JSON file containing model architecture in a format Keras understands.             |
| `model.h5`                   | Model weights.                                                                     |
| `weights_logger_callback.py` | Implements a Keras callback that keeps track of model weights throughout training. |
| `drive.py`                   | Implements driving simulator callbacks, essentially communicates with the driving simulator app providing model predictions based on real-time data simulator app is sending. |
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExNjIyMTE4MjAsLTEwNDgwODE0OV19
-->