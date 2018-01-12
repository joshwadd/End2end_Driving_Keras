---


---

<h1 id="end-2-end-autonomous-driving-in-keras">End 2 End Autonomous Driving in Keras</h1>
<p>In this project, I use convolutional deep neural networks to clone driving behavior by training end to end from input camera images to output steering command to an autonomous vehicle. Keras is using for training, validating and testing the model.</p>
<p>The Udacity Self-Driving Car simulator was used for acquiring training data sets of human driving behavior around test tracks. The convolutional neural network was then trained to map input images to steering angles as a regression problem. Once the model has learnt these mappings arising from human behavior it can be used to generate new steering angles online to control the autonomous vehicle in the simulator.</p>
<h2 id="project-files">Project Files</h2>

<table>
<thead>
<tr>
<th>File</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>data.py</code></td>
<td>Methods related to data augmentation, preprocessing and batching.</td>
</tr>
<tr>
<td><code>model.py</code></td>
<td>Implements model architecture and runs the training pipeline.</td>
</tr>
<tr>
<td><code>model.json</code></td>
<td>JSON file containing model architecture in a format Keras understands.</td>
</tr>
<tr>
<td><code>model.h5</code></td>
<td>Model weights.</td>
</tr>
<tr>
<td><code>weights_logger_callback.py</code></td>
<td>Implements a Keras callback that keeps track of model weights throughout training.</td>
</tr>
<tr>
<td><code>drive.py</code></td>
<td>Implements driving simulator callbacks, essentially communicates with the driving simulator app providing model predictions based on real-time data simulator app is sending.</td>
</tr>
</tbody>
</table>
