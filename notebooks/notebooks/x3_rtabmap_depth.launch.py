from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='rtabmap_sync',
            executable='rgbd_sync',
            name='rgbd_sync',
            output='screen',
            parameters=[{
                'approx_sync': True,
                'approx_sync_max_interval': 0.02,
                'queue_size': 50
            }],
            remappings=[
                ('rgb/image', '/color/image_raw'),
                ('depth/image', '/depth/image_raw'),
                ('rgb/camera_info', '/color/camera_info'),
                ('rgbd_image', '/rgbd_image')
            ]
        ),

        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='rgbd_odometry',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'odom_frame_id': 'odom',
                'publish_tf': True,
                'approx_sync': True,
                'approx_sync_max_interval': 0.02,
                'queue_size': 50,
                'Vis/MinInliers': '8',
                'Odom/ResetCountdown': '0'
            }],
            remappings=[
                ('rgb/image', '/color/image_raw'),
                ('depth/image', '/depth/image_raw'),
                ('rgb/camera_info', '/color/camera_info')
            ]
        ),

        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'subscribe_rgbd': True,
                'subscribe_depth': False,
                'subscribe_rgb': False,
                'subscribe_scan': False,
                'subscribe_scan_cloud': False,
                'approx_sync': True,
                'queue_size': 50,
                'Grid/3D': True,
                'Grid/FromDepth': True,
                'RGBD/CreateOccupancyGrid': True,
                'Reg/Force3DoF': 'false'
            }],
            remappings=[
                ('rgbd_image', '/rgbd_image'),
                ('odom', '/odom')
            ],
            arguments=['-d']
        ),

        Node(
            package='rtabmap_viz',
            executable='rtabmap_viz',
            name='rtabmap_viz',
            output='screen',
            parameters=[{
                'frame_id': 'base_footprint',
                'subscribe_rgbd': True,
                'subscribe_depth': False,
                'subscribe_rgb': False,
                'subscribe_scan': False,
                'approx_sync': True
            }],
            remappings=[
                ('rgbd_image', '/rgbd_image'),
                ('odom', '/odom')
            ]
        ),
    ])