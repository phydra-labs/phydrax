# ROS bag admission

`RosbagImportPlan` decodes only an explicit topic and message-type allow-list under message and byte limits. It preserves sensor and bag timestamps separately, topic sequence, clock identity, and out-of-order evidence.

Supported profiles lower images, camera calibration arrays, point clouds, planar laser scans, IMU, odometry, joint state, and dynamic/static transforms into typed assets and frame timelines. Unsupported topics are not decoded implicitly.

The optional host adapter uses no live ROS runtime. JAX execution receives only normalized measurement fields and prepared frame/clock routes.
