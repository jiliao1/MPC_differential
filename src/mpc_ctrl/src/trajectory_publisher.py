#!/usr/bin/env python3  
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
from scipy.interpolate import CubicSpline

def generate_bspline_path(path, sampling_distance):
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    total_distance = cumulative_distances[-1]
    num_samples = int(total_distance / sampling_distance)
    spline_x = CubicSpline(cumulative_distances, path[:, 0])
    spline_y = CubicSpline(cumulative_distances, path[:, 1])
    sampled_distances = np.linspace(0, total_distance, num_samples)
    sampled_x = spline_x(sampled_distances)
    sampled_y = spline_y(sampled_distances)

    return np.vstack((sampled_x, sampled_y)).T, sampled_distances

def generate_circle_path(radius=10, center=(0, 1), num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.vstack((x, y)).T

def generate_eight_path(radius=10, num_points=100):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.sin(t)
    y = radius * np.sin(2 * t)
    return np.vstack((x, y)).T

def calculate_orientation(path, distances):
    # Calculate the orientation based on the derivative of the path (tangent vector)
    dx = np.gradient(path[:, 0], distances)
    dy = np.gradient(path[:, 1], distances)
    angles = np.arctan2(dy, dx)
    return angles

def publish_trajectory(path_type='circle', radius=1, sampling_distance=1):
    rospy.init_node('trajectory_publisher')
    pub = rospy.Publisher('/trajectory', Path, queue_size=10)
    
    if path_type == 'circle':
        path = generate_circle_path(radius=radius)
    elif path_type == 'eight':
        path = generate_eight_path(radius=radius)
    else:
        rospy.logerr("Invalid path type!")
        return

    smooth_path, distances = generate_bspline_path(path, sampling_distance)
    orientations = calculate_orientation(smooth_path, distances)

    path_msg = Path()
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = 'odom'

    for i, point in enumerate(smooth_path):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = 0.0

        # Set orientation (quaternion from yaw angle)
        yaw = orientations[i]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]

        path_msg.poses.append(pose)

    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        path_msg.header.stamp = rospy.Time.now()
        pub.publish(path_msg)
        rate.sleep()

if __name__ == '__main__':

    try:
        publish_trajectory(path_type='eight', radius=1, sampling_distance=0.01)
    except rospy.ROSInterruptException:
        pass
