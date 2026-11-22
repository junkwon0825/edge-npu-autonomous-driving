import carla
import random
import time
import numpy as np
import cv2
from queue import Queue, Empty

#1. data
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
traffic_manager = client.get_trafficmanager(8000)

original_settings = world.get_settings()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager.set_synchronous_mode(True)

blueprints = world.get_blueprint_library()

vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
lidar_bp = blueprints.find("sensor.lidar.ray_cast")

lidar_bp.set_attribute("channels", "32")
lidar_bp.set_attribute("range", "50")
lidar_bp.set_attribute("points_per_second", "100000")
lidar_bp.set_attribute("rotation_frequency", "20")
lidar_bp.set_attribute("sensor_tick", "0.05")

spawn_points = world.get_map().get_spawn_points()

#2. model
vehicle = world.spawn_actor(
    vehicle_bp,
    random.choice(spawn_points)
)

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)

lidar_transform = carla.Transform(
    carla.Location(x=0.0, z=2.2)
)

lidar = world.spawn_actor(
    lidar_bp,
    lidar_transform,
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

lidar_queue = Queue()
spectator = world.get_spectator()

#3. compile,train
lidar.listen(lidar_queue.put)

#4. evaluate,predict
print("LiDAR visualization ON")

try:
    while True:
        world.tick()

        try:
            lidar_data = lidar_queue.get(timeout=2.0)

        except Empty:
            print("LiDAR timeout")
            continue

        points = np.frombuffer(
            lidar_data.raw_data,
            dtype=np.float32
        )

        points = np.reshape(
            points,
            (-1, 4)
        )

        xy = points[:, :2]

        image_size = 800
        scale = 7

        lidar_image = np.zeros(
            (image_size, image_size, 3),
            dtype=np.uint8
        )

        x = (xy[:, 0] * scale + image_size / 2).astype(np.int32)
        y = (-xy[:, 1] * scale + image_size / 2).astype(np.int32)

        mask = (
            (x >= 0) &
            (x < image_size) &
            (y >= 0) &
            (y < image_size)
        )

        lidar_image[
            y[mask],
            x[mask]
        ] = 255

        cv2.circle(
            lidar_image,
            (image_size // 2, image_size // 2),
            5,
            (0, 0, 255),
            -1
        )

        cv2.imshow(
            "LiDAR BEV",
            lidar_image
        )

        transform = vehicle.get_transform()

        spectator.set_transform(
            carla.Transform(
                transform.location
                + carla.Location(z=3.0)
                - transform.get_forward_vector() * 8.0,
                carla.Rotation(
                    pitch=-15.0,
                    yaw=transform.rotation.yaw
                )
            )
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    lidar.stop()
    lidar.destroy()
    vehicle.destroy()

    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(original_settings)

    cv2.destroyAllWindows()

    print("Actors destroyed")
    print("World settings restored")