#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>

#include <ros/ros.h>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include "qp.hpp"

#define PI 3.14159265359
ros::Publisher odom_path_pub, cmd_vel_pub, transformed_path_pub;

ros::Subscriber odom_sub, refer_path_sub;
nav_msgs::Path odom_path;
std::vector<geometry_msgs::PoseStamped> original_path;

constexpr unsigned short STATE_NUM = 3;
constexpr unsigned short CTRL_NUM = 2;
constexpr unsigned short MPC_WINDOW = 40;

std::vector<Eigen::Matrix<double, STATE_NUM, 1>> xref;
Eigen::Matrix<double, STATE_NUM, 1> x0;

static int findClosestPathIndex(const geometry_msgs::Pose &odom_pose)
{
	double min_distance = std::numeric_limits<double>::max();
	int closest_index = 0;
	for (size_t i = 0; i < original_path.size(); ++i)
	{
		double dx = original_path[i].pose.position.x - odom_pose.position.x;
		double dy = original_path[i].pose.position.y - odom_pose.position.y;
		double distance = std::sqrt(dx * dx + dy * dy);

		if (distance < min_distance)
		{
			min_distance = distance;
			closest_index = i;
		}
	}
	return closest_index;
}
static void pathCallback(const nav_msgs::Path::ConstPtr &path_msg)
{
	original_path = path_msg->poses;
}
void convertPathToXRef(
	const nav_msgs::Path &transformed_path)
{
	xref.clear();

	for (const auto &pose : transformed_path.poses)
	{
		Eigen::Matrix<double, STATE_NUM, 1> state;

		state(0) = pose.pose.position.x;
		state(1) = pose.pose.position.y;
		state(2) = tf::getYaw(pose.pose.orientation);
		;
		xref.push_back(state);
	}
}
void odomCallback(const nav_msgs::Odometry::ConstPtr &odom_msg, tf2_ros::Buffer &tf_buffer)
{
	if (original_path.empty())
	{
		ROS_WARN("No path received yet.");
		return;
	}

	try
	{
		auto transform = tf_buffer.lookupTransform(

			odom_msg->child_frame_id,
			odom_msg->header.frame_id,
			ros::Time(0),
			ros::Duration(0.1));

		int closest_index = findClosestPathIndex(odom_msg->pose.pose);

		std::vector<geometry_msgs::PoseStamped> selected_points;
		for (int i = 0; i < MPC_WINDOW + 1; ++i)
		{
			int index = std::min(closest_index + i, static_cast<int>(original_path.size() - 1));
			selected_points.push_back(original_path[index]);
		}

		nav_msgs::Path transformed_path;
		transformed_path.header.frame_id = odom_msg->child_frame_id;
		transformed_path.header.stamp = ros::Time::now();
		for (const auto &pose : selected_points)
		{
			geometry_msgs::PoseStamped transformed_pose;
			tf2::doTransform(pose, transformed_pose, transform);
			transformed_path.poses.push_back(transformed_pose);
		}
		convertPathToXRef(transformed_path);
		transformed_path_pub.publish(transformed_path);
	}
	catch (tf2::TransformException &ex)
	{
		ROS_WARN("Transform unavailable: %s", ex.what());
	}

	geometry_msgs::PoseStamped odom_pose;
	odom_pose.header = odom_msg->header;
	odom_pose.pose = odom_msg->pose.pose;
	odom_path.poses.push_back(odom_pose);
	odom_path.header.frame_id = "odom";
	odom_path.header.stamp = ros::Time::now();
	odom_path_pub.publish(odom_path);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "mpc_node");
	ros::NodeHandle n;
	tf2_ros::Buffer tf_buffer;
	tf2_ros::TransformListener tf_listener(tf_buffer);

	cmd_vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 100);
	transformed_path_pub = n.advertise<nav_msgs::Path>("/transformed_path", 100);
	odom_path_pub = n.advertise<nav_msgs::Path>("/odom_path", 100);

	refer_path_sub = n.subscribe<nav_msgs::Path>("/trajectory", 100, pathCallback);
	odom_sub = n.subscribe<nav_msgs::Odometry>(
		"/odom", 100, boost::bind(odomCallback, _1, boost::ref(tf_buffer)));

	Eigen::Matrix<double, STATE_NUM, 1> xMax;
	Eigen::Matrix<double, STATE_NUM, 1> xMin;
	Eigen::Matrix<double, CTRL_NUM, 1> uMax;
	Eigen::Matrix<double, CTRL_NUM, 1> uMin;
	Eigen::DiagonalMatrix<double, STATE_NUM> Q;
	Eigen::DiagonalMatrix<double, CTRL_NUM> R;

	uMax << 1, PI;
	uMin << -1, -PI;

	xMax << OsqpEigen::INFTY, OsqpEigen::INFTY, PI;
	xMin << -OsqpEigen::INFTY, -OsqpEigen::INFTY, -PI;

	Q.diagonal() << 1000,1000,500;
	R.diagonal() << 10, 1;
	x0 << 0, 0, 0;
	Eigen::VectorXd out;
	out.resize(2);
	out << 0, 0;
	ros::Rate loop_rate(50);
	int count = 0;
	while (ros::ok())
	{
		if (xref.size() > 0)
		{
			MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW> MPC_Solver(Q, R, xMax, xMin, uMax, uMin);
			MPC_Solver.set_x_xref(x0, out, xref);
			out = MPC_Solver.Solver();
			geometry_msgs::Twist vel_msg;
			vel_msg.linear.x = out(0);
			vel_msg.angular.z = out(1);
			cmd_vel_pub.publish(vel_msg);
			ROS_INFO("[%0.2f m/s, %0.2f rad/s]",
					 vel_msg.linear.x, vel_msg.angular.z);
		}
		loop_rate.sleep();
		ros::spinOnce();
	}
}