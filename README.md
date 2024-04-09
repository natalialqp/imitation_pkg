<!--
* Graph-Based Learning from Demonstration Framework
*
* Copyright (c) 2024 Natalia Quiroga Perez. All rights reserved.
*
* This software may be modified and distributed
* under the terms of the BSD 3-Clause license.
*
* Refer to the LICENSE file for details.
*
-->

<h1>Graph-Based Learning from Demonstration Framework<sup>®</sup></h1>

<a id="markdown-description" name="description"></a>
# Description

The official repository contains documentation and datasets for QTrobot, NAO and Kinova gen3. This Framework runs in Python.
The repository has been tested on Ubuntu 20.04.

<h1>Table of Contents</h1>

<!-- TOC -->

- [Description](#description)
- [Quick Start](#quickstart)
- [Reference](#reference)
      - [Useful Links](#useful-links)

<a id="description" name="description"></a>
<h1> Description</h1>

This repository stores the learning from demonstration framework. It focuses on the upper-body limb for the robots QTrobot, NAO and Freddy (2 Kinova arms Gen3 on a Kelo base).

You are going to find main code in the file main.py. The one has different options to perform different steps of learning from demonstration.

The results are placed in the folder data. Where the sub-folder test_qt, test_nao and test_gen3 contain self-exploration files, objects ,paths, graphs and trajectory libraries.

For the main file:
<li> Add the name of the robot (qt, nao or gen3)</li>
<li> Define the amount of babbling points (30, 100 or 150)</li>
<li> Select the corresponding flag to run the desired function: </li>

<ul>
<li> explore-world: to interpolate points between collected babbling points </li>
<li> pose-predicition: to iterate over the 12 pre-recorded actions and increase the graphs reachability </li>
<li> path-planning: to perform path planning for the pre-recorded actions </li>
<li> object-in-robot-graph: to test the objects in the world </li>
<li> read-library-paths: to read a trajectory from the trajectory library </li>
<li> create-object: to create an object file </li>
</ul>

<li>For the GMR.py:</li>
<ul>
      <li>  Add the name of the robot (qt, nao or gen3) </li>
      <li>  Define the amount of babbling points (30, 100 or 150)</li>
      <li>  Select the action class</li>
      <li>  Select the end_effector_dict from the yaml file of the corresponding robot</li>
      <li>  Select the number of components for the GMM for sklearn </li>
      <li>  In GMPlotter.py select the number of components for the gmr test_gmr(...., num_components)</li>
</ul>
<li>For motor-babbling: select the amount of points to be collected and the delta angle to babble  (default = 10)</li>
<ul>
      <li>  QTrobot: run the subscribe_node.py file</li>
      <li>  NAO: select the option motor-babbling in nao_controller.py</li>
      <li>  Freddy: select the option motor-babbling in freddy_controller.py</li>
</ul>
<li>For action reproduction on the robots:</li>
<ul>
      <li>  QTrobot: run the qt_action_player.py file</li>
      <li>  NAO: select the option reproduce-action in nao_controller.py</li>
      <li>  Freddy: select the option reproduce-action in freddy_controller.py</li>
 </ul>

<a id="quickstart" name="quickstart"></a>
<h1> Quick Start </h1>

  To run the Python examples you will need to install the Python3 and run the requirements.txt file</li>
      <li>  QTrobot: uses ROS Noetic, make sure to connect to the robot following the LuxAI documentation</li>
      <li>  NAO: runs on Ubuntu 16, make sure to have python 2.7 an Choregraphe installed </li>
      <li>  Freddy: clone the kortex repository outside of the imitation_pkg and copy the required files, follow the instructions in freddy_controller.py</li>


<h1> Reference </h1>
#### Useful Links

|  |  |
| ---: | --- |
| QTrobot quick start: | [https://docs.luxai.com/docs/intro_code](https://docs.luxai.com/docs/intro_code) |
| Kinova kortex: | [https://www.kinovarobotics.com](https://github.com/Kinovarobotics/kortex)|
| NAO documentation: | [http://doc.aldebaran.com/2-8/family/nao_technical/index_naov6.html](http://doc.aldebaran.com/2-8/family/nao_technical/index_naov6.html)|
| GMR repository: | [https://github.com/AlexanderFabisch/gmr](https://github.com/AlexanderFabisch/gmr)|
