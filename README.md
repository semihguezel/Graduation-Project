# Graduation-Project
Position and Orientation Estimation using 360° spherical camera

# Project Introduction
&emsp; 360° spherical cameras have recently started to be preferred more frequently due to the ease of use, low cost, and the development of image processing technology. The frequency of use in fields such as robotics and Virtual Reality has increased. In addition, 360° spherical cameras are also preferred in indoor positioning systems where satellite technologies do not provide precise results. In this method, high accuracy position information can be obtained using 360° spherical cameras.

# Goals & Objectives of the Project
* Development of object tracking algorithm
* Estimating bearing angle with 2° accuracy
* Obtaining location information with sub-meter accuracy
* Development of map user interface for monitoring purposes

# Mathematical Method
&emsp; Position and orientation can be obtained from spherical cameras with the epipolar geometry method. Imagine a 10x10 room with the reference point located at x0, y0. Let's assume that the position information of the uniformly distributed markers in the room is obtained from a spherical camera will be placed at this reference point. Then, a spherical camera with a distance T from our reference point is placed in the room. In this case, using the epipolar geometry method, it is possible to determine where the positions of the markers in the image obtained from the spherical camera at the reference point are in the spherical camera recently placed in the room.
  
<p>
  <img src="https://github.com/semihguezel/Graduation-Project/blob/Main/images/Epipolar_Geometry.png" width="350" title="Epipolar Geometry">
</p>

# Estimating Rotation and Transformation Matrices
&emsp; In epipolar geometry, estimation of Rotation & Transformation matrices can be computed by decomposition of Essential matrix. However in our project we decompose these matrices by using BFGS optimizer algorithm.

# BFGS Optimization Algorithm
&emsp; The BFGS algorithm is one of the Quasi-Newton methods. In Quasi-Newton methods, the hessian matrix is produced by taking the partial second derivatives of the given inputs. However, it has a limitation as it requires the calculation of the inverse of the Hessian that can be computationally intensive. The Quasi-Newton method approximates the inverse of the Hessian using the gradient and hence can be computationally feasible.The BFGS method (the L-BFGS is an extension of BFGS) updates the calculation of the Hessian matrix at each iteration rather than recalculating it [1]. In our project, the BFGS algorithm minimizes the projection errors in the transition from world coordinates to camera coordinates.
  
# Test Enviroment
&emsp; Webots software, a 3D robot simulator, was chosen to set up the necessary environment because a 10x10 meter room and an autonomously moving robot in this room are needed to solve the problem. The factory environment provided by the Webots simulator by default was imported into the simulator, and then the dimensions of the factory were resized to be a 10x10 square meter room. In the next step, a predetermined number of rectangular markers to be placed on the wall. Finally, 360° footage of the test enviroment were converted into Equirectangular projection.

<p>
  <img src="https://github.com/semihguezel/Graduation-Project/blob/Main/images/Equirectangular.png" width="500" title="Equirectangular Projection">
</p>

# Marker Tracking Algorithm
&emsp; Classic object detection algorithms are not suitable for our project. This is because we work with a 360° camera. Since the images taken from the camera we use are Equirectangular projection, some parts of the image are distorted. Object detection algorithms produced in accordance with perspective projections where distortions occur completely fails at the marker detection stage. In order to eliminate this problem, equirectangular images taken from the camera were converted to Cube Map projection. In this way, the distortions due to the very wide viewing angle on the image were eliminated.

<p>
  <img src="https://github.com/semihguezel/Graduation-Project/blob/Main/images/equi_to_cubemap.png" width="700" height = "200" title="Equirectangular to Cubemap Projection">
</p>

&emsp; In order to calculate the coordinates and rotation of the vehicle in the room, the markers must be continuously detected in each frame of the videos to be used. If a center pixel coordinate cannot be obtained for each marker used in each frame, the calculation of the vehicle's position will fail. In the images taken while the vehicle is in motion, the coordinates of the markers on the image change. Therefore, the tracker algorithm to be used should be able to accurately calculate the change on the image and track the markers in every frame. In this context, it has been decided to use the trackers in the OpenCV library. Six of the tracker algorithms in the OpenCV library were selected and trials were conducted with each of them. These trackers are: Boosting Tracker, MIL Tracker, KCF Tracker, CSRT Tracker, MedianFlow Tracker and Mosse Tracker. We chose to use the CSRT tracker algorithm to track markers in the image converted to cube map format. This is because other trackers cannot provide 20 cm position accuracy and 2 degrees rotation accuracy, which are the limitations of the project.

<p>
  <img src="https://github.com/semihguezel/Graduation-Project/blob/Main/images/Marker_Tracking.png" width="500" title="Marker Tracking Algorithm">
</p>

# Map User Interface
&emsp; We used Unity physics engine to develop Map UI. Since our Position & Orientation estimation algorithm developed in python and Unity uses C# as a backend we could not transfer the data directly. So we were established a communication where we can pass the computed arguments in python to Unity by using the Socket communication protocol. 

<p>
  <img src="https://github.com/semihguezel/Graduation-Project/blob/Main/images/Map_UI.png" width="700" title="Map User Interface">
</p>

# Results
&emsp; We created several scenarios in the Webots environment to test our algorithm and analyze the results. Among these scenarios, the U-route scenario was examined and the test results were shown in table and graph.
<p>
  <img src="https://github.com/semihguezel/Graduation-Project/blob/Main/images/Error_Table.png" width="500" title="Error Table"><br>
  <h> 
    Test results for U route scenario
  </h>
</p></br>
<p>
 <img src="https://github.com/semihguezel/Graduation-Project/blob/Main/images/Error_Graph.png" width="1000" height="500" title="Error Graph">
  <h> 
    Errors of X, Y, Z axis both coordinates and rotation for each frame  
  </h>
</p>
As seen in the results, both location and routing errors are within the scope of the project.

# How to use Tracking Algorithm
&emsp; I provided necessary codes and materials to run our algorithm, in tracking_algorithm folder. In order to run tracking algorithm, run tracker.py provided in these files. You have to change the current path of the example video, named ublue in videos folder for obtaining results. When algorithm started just select the blue frames of the markers located in the room. After that, just wait for algorithm to estimate position & orientation of the AGV for given video. Test results will be saved in txt folder for both BFGS and L-BFGSL methods. You can run rmse script for obtaining erros.
# Map UI
&emsp; You can check my other repository for the Map User Interface I created in Unity environment. I have provided Unity files and python link script in this repository. To use them, you need to run the simulation and the python connection script simultaneously. At first you will see an Input Scene, you have to type your AGV's initial position in float format. In order to observe the position and rotation of the vehicle, you need to load the position and rotation information obtained from the tracking algorithm in the calculate_pos() function of python connection script.<br>
https://github.com/semihguezel/Map-User-Interface

# Project Report
You can view the report of our project as a pdf file here.<br>
https://github.com/semihguezel/Graduation-Project/blob/Main/211215_GraduationProjectReport.pdf

# References
1. https://towardsdatascience.com/numerical-optimization-based-on-the-l-bfgs-method-f6582135b0ca

