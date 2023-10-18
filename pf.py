import math
import rospy
import copy
import numpy as np
import tf
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from nav_msgs.msg import Odometry
from pf_base import PFLocaliserBase
from util import rotateQuaternion, getHeading
from random import random, gauss, vonmisesvariate


class PFLocaliser(PFLocaliserBase):  # Define the PFLocaliser class, which inherits from PFLocaliserBase.

    def __init__(self):  # Constructor method for the class.
        super(PFLocaliser, self).__init__()  # Call the constructor of the superclass (PFLocaliserBase) to ensure proper initialization.

        # Set motion model parameters.

        self.ODOM_ROTATION_NOISE = 0.02  # Define the amount of rotation noise in the odometry model.
        self.ODOM_TRANSLATION_NOISE = 0.02  # Define the noise level for forward translation in the odometry model.
        self.ODOM_DRIFT_NOISE = 0.02  # Define the noise level for side-to-side movement (drift) in the odometry model.

        # Sensor model parameters.

        self.NUMBER_PREDICTED_READINGS = 50  # Set the number of sensor readings that we aim to predict.

        # Set motion model parameters to add noise while resampling.

        self.UPDATED_NOISE = np.random.uniform(100, 120)  # Initialize the updated noise with a random value between 100 and 120.
        self.UPDATED_ANGULAR_NOISE = np.random.uniform(1, 120)  # Initialize the angular noise with a random value between 1 and 120.

    def obtained_value(self, prob_array):  # Define a function to obtain values based on a provided probability array.

        pose_arrays = PoseArray()  # Create an empty PoseArray object to store selected poses based on their weights.

        # Iterate through each pose in the particle cloud.

        for each_pose in range(len(self.particlecloud.poses)):

            # Calculate a threshold value by multiplying a random number between 0 and 1 with the sum of all probabilities.

            total_value = random.random() * sum(prob_array)
            total_weight_probability = 0  # Initialize a counter to sum up the weights of the particles.
            indicator = 0  # Initialize an index variable to track the particle being considered.

            # Loop until the summed weight exceeds the calculated threshold.

            while total_weight_probability < total_value:
                total_weight_probability += prob_array[indicator]  # Add the weight of the current particle to the total weight.
                indicator = indicator + 1  # Move to the next particle.

            # Append the pose that passed the threshold to the new pose array.

            pose_arrays.poses.append(copy.deepcopy(self.particlecloud.poses[indicator
                    - 1]))

        return pose_arrays  # Return the new pose array containing the selected poses.

    def updated_noise(self, pose_object):  # Define a function to add noise to a given pose.

        # Check if the updated noise value is greater than 1.0.

        if self.UPDATED_NOISE > 1.0:
            self.UPDATED_NOISE -= 0.02  # If true, reduce the noise by 0.02.
        else:

            # Otherwise, reinitialize the noise with a random value between 0.02 and 1.

            self.UPDATED_NOISE = np.random.uniform(0.02, 1)

        # Randomly set new noise levels for the motion model parameters.

        self.ODOM_ROTATION_NOISE = np.random.uniform(0.02, 0.1)  # Rotation noise.
        self.ODOM_TRANSLATION_NOISE = np.random.uniform(0.02, 0.1)  # Forward translation noise.
        self.ODOM_DRIFT_NOISE = np.random.uniform(0.02, 0.1)  # Side-to-side movement noise.

        # Apply noise to the pose object's position.

        pose_object.position.x += gauss(0, self.UPDATED_NOISE) \
            * self.ODOM_TRANSLATION_NOISE  # Add noise to the X position.
        pose_object.position.y += gauss(0, self.UPDATED_NOISE) \
            * self.ODOM_DRIFT_NOISE  # Add noise to the Y position.

        # Apply noise to the pose object's orientation.

        angle_noise = (vonmisesvariate(0, self.UPDATED_ANGULAR_NOISE)
                       - math.pi) * self.ODOM_ROTATION_NOISE
        pose_object.orientation = \
            rotateQuaternion(pose_object.orientation, angle_noise)

        return pose_object  # Return the pose object after adding noise.


def initialise_particle_cloud(self, initialpose):

    # Initialize a PoseArray object for storing particles.

    pose_arrays = PoseArray()

    # Initialize an index to count up to 500 particles.

    i = 0

    # Populate the PoseArray with 500 particles.

    while i < 500:

        # Sample a Gaussian random number with mean 0 and standard deviation 1.

        random_gauss_number = gauss(0, 1)

        # Sample a value from a von Mises distribution (similar to a Gaussian distribution on a circle).

        rotational_dist = vonmisesvariate(0, 5)

        # Create a new Pose object for the particle.

        pose_objects = Pose()

        # Initialize the particle's position with noise added to the initial pose.

        pose_objects.position.x = initialpose.pose.pose.position.x \
            + random_gauss_number * self.ODOM_TRANSLATION_NOISE
        pose_objects.position.y = initialpose.pose.pose.position.y \
            + random_gauss_number * self.ODOM_DRIFT_NOISE

        # Initialize the particle's orientation with noise added to the initial orientation.

        noise = (rotational_dist - math.pi) * self.ODOM_ROTATION_NOISE
        pose_objects.orientation = \
            rotateQuaternion(initialpose.pose.pose.orientation, noise)

        # Append the initialized particle to the PoseArray.

        pose_arrays.poses.append(pose_objects)

        # Increment the index.

        i += 1

    # Return the PoseArray populated with particles.

    return pose_arrays


def update_particle_cloud(self, scan):

    # Initialize a list to store the weights (probabilities) of the particles.

    prob_of_weight = []

    # Iterate through each particle in the particle cloud.

    for pose_object in self.particlecloud.poses:

        # Compute the weight of the particle based on a sensor model and the current scan.

        prob_of_weight.append(self.sensor_model.get_weight(scan,
                              pose_object))

    # Resample the particle cloud based on the computed weights.

    obtained_pose_arrays = self.obtained_value(prob_of_weight)

    # Update each resampled particle's pose by adding noise.

    for pose_object in obtained_pose_arrays.poses:
        pose_object = self.updated_noise(pose_object)

    # Update the class's particle cloud with the resampled particles.

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

    # Convert the quaternion representation of orientation to roll, pitch, yaw angles.

    orientation_list = [predicted_pose.position.x,
                        predicted_pose.position.y,
                        predicted_pose.orientation.z,
                        predicted_pose.orientation.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

    # Print the estimated robot's position and heading.

    print ('Robots X position: ', predicted_pose.position.x)
    print ('Robots Y position: ', predicted_pose.position.y)
    print ('Robots Heading: ', math.degrees(yaw))

    # Return the estimated robot's pose.

    return predicted_pose

