
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from nav_msgs.msg import Odometry
from . pf_base import PFLocaliserBase
import math
import rospy
from . util import rotateQuaternion, getHeading
from random import random
from time import time
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import random
from random import gauss, vonmisesvariate
from numpy import random
import numpy as np
import copy
import tf


class PFLocaliser(PFLocaliserBase):
       
    def __init__(self):
        # Call the superclass constructor
        super(PFLocaliser, self).__init__()
        
        # Set motion model parameters
        self.ODOM_ROTATION_NOISE = 0.02 # Odometry model rotation noise
        self.ODOM_TRANSLATION_NOISE = 0.02 # Odometry x axis (forward) noise
        self.ODOM_DRIFT_NOISE = 0.02 # Odometry y axis (side-side) noise
 
        # Sensor model parameters
        self.NUMBER_PREDICTED_READINGS = 50 	# Number of readings to predict
        
        # Set motion model paramaters to add noise while resampling
        self.UPDATED_NOISE = np.random.uniform(100, 120) 
        self.UPDATED_ANGULAR_NOISE = np.random.uniform(1, 120) 
    
    
    def obtained_value(self, prob_array): # Define a function to obtain values based on a provided probability array.
        
        pose_arrays = PoseArray()  # Create an empty PoseArray object to store selected poses based on their weights.
        # Iterate through each pose in the particle cloud.
        for each_pose in range(len(self.particlecloud.poses)):
        # Calculate a threshold value by multiplying a random number between 0 and 1 with the sum of all probabilities.
            total_value = random.random() * sum(prob_array)
            total_weight_probability = 0  # Initialize a counter to sum up the weights of the particles.
            indicator = 0
            # Loop until the summed weight exceeds the calculated threshold.
            while total_weight_probability < total_value:
	            total_weight_probability += prob_array[indicator] # Add the weight of the current particle to the total weight
	            indicator = indicator + 1  # Move to the next particle.
	            
	            # Append the pose that passed the threshold to the new pose array.
            pose_arrays.poses.append(copy.deepcopy(self.particlecloud.poses[indicator - 1])) 
        return pose_arrays 
      
    # Return the new pose array containing the selected poses.
    
    # Define a function to add noise to a given pose.
        
    def updated_noise(self, pose_object):
        # Check if the updated noise value is greater than 1.0.
        if self.UPDATED_NOISE > 1.0: 
            self.UPDATED_NOISE -= 0.02
        else:
            self.UPDATED_NOISE = np.random.uniform(0.02, 1) 
        # Randomly set new noise levels for the motion model parameters.
        self.ODOM_ROTATION_NOISE = np.random.uniform(0.02, 0.1)
        self.ODOM_TRANSLATION_NOISE = np.random.uniform(0.02, 0.1)
        self.ODOM_DRIFT_NOISE = np.random.uniform(0.02, 0.1)

# Apply noise to the pose object's position.
        pose_object.position.x += gauss(0, self.UPDATED_NOISE) * self.ODOM_TRANSLATION_NOISE   
        pose_object.position.y += gauss(0, self.UPDATED_NOISE) * self.ODOM_DRIFT_NOISE   
        pose_object.orientation = rotateQuaternion(pose_object.orientation,(vonmisesvariate(0, self.UPDATED_ANGULAR_NOISE) - math.pi) * self.ODOM_ROTATION_NOISE) 
        return pose_object
        
        
    def initialise_particle_cloud(self, initialpose):

        pose_arrays = PoseArray()
        i = 0
        while i < 500:
            random_gauss_number = gauss(0, 1)
            rotational_dist = vonmisesvariate(0, 5)
            pose_objects = Pose()
            pose_objects.position.x = initialpose.pose.pose.position.x + (random_gauss_number * self.ODOM_TRANSLATION_NOISE)
            pose_objects.position.y = initialpose.pose.pose.position.y + (random_gauss_number * self.ODOM_DRIFT_NOISE)
            pose_objects.orientation = rotateQuaternion(initialpose.pose.pose.orientation,((rotational_dist - math.pi) * self.ODOM_ROTATION_NOISE))
            pose_arrays.poses.append(pose_objects)
            i += 1
        return pose_arrays
# Return the PoseArray populated with particles.

    def update_particle_cloud(self, scan):
        
        prob_of_weight = []
        
        for pose_object in self.particlecloud.poses:
            prob_of_weight.append(self.sensor_model.get_weight(scan, pose_object))

        obtained_pose_arrays = self.obtained_value(prob_of_weight)
                
        for pose_object in obtained_pose_arrays.poses:
            pose_object = self.updated_noise(pose_object)
         
        self.particlecloud = obtained_pose_arrays
 
    
    def estimate_pose(self):
        # Initialize a Pose object to store the estimated robot's pose.

        predicted_pose = Pose()
        
            # Initialize variables to accumulate the positions and orientations of all particles.

        x_value = 0
        y_value = 0
        z_value = 0  
        w_value = 0  
            # Sum the positions and orientations of all particles.

        for each_object in self.particlecloud.poses:
            x_value += each_object.position.x 
            y_value += each_object.position.y 
            z_value += each_object.orientation.z 
            w_value += each_object.orientation.w 

    # Compute the average position and orientation to get the estimated robot's pose.

        predicted_pose.position.x = x_value / 500
        predicted_pose.position.y = y_value / 500
        predicted_pose.orientation.z = z_value / 500
        predicted_pose.orientation.w = w_value / 500

        #rpy = tf.transformations.euler_from_quaternion(predicted_pose.position.x , predicted_pose.position.y, predicted_pose.orientation.z, predicted_pose.orientation.w)
        orientation_list = [predicted_pose.position.x, predicted_pose.position.y, predicted_pose.orientation.z, predicted_pose.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        
            # Print the estimated robot's position and heading.

        print('Robots X position: ', predicted_pose.position.x)
        print('Robots Y position: ', predicted_pose.position.y)
        print('Robots Heading: ', math.degrees(yaw))
            # Return the estimated robot's pose.

        return predicted_pose
