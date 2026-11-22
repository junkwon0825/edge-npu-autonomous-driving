import carla
import random
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
camera_bp = blueprints.find("sensor.camera.rgb")
lidar_bp = blueprints.find("sensor.lidar.ray_cast")

camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "450")
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", "0.05")

lidar_bp.set_attribute("channels", "32")
lidar_bp.set_attribute("range", "50")
lidar_bp.set_attribute("points_per_second", "100000")
lidar_bp.set_attribute("rotation_frequency", "20")
lidar_bp.set_attribute("sensor_tick", "0.05")

spawn_points = world.get_map().get_spawn_points()

#2. model
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
vehicle.set_autopilot(True, traffic_manager.get_port())

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(carla.Location(x=1.5, z=1.7)),
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

lidar = world.spawn_actor(
    lidar_bp,
    carla.Transform(carla.Location(x=0.0, z=2.2)),
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

camera_queue = Queue()
lidar_queue = Queue()

spectator = world.get_spectator()

#3. compile,train
camera.listen(camera_queue.put)
lidar.listen(lidar_queue.put)

#4. evaluate,predict
print("Sensor dashboard ON")

try:
    while True:
        world.tick()

        try:
            image = camera_queue.get(timeout=2.0)
            lidar_data = lidar_queue.get(timeout=2.0)

        except Empty:
            print("Sensor timeout")
            continue

        image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_array = np.reshape(image_array, (image.height, image.width, 4))
        rgb_image = image_array[:, :, :3]

        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4))

        xy = points[:, :2]

        bev_size = 450
        scale = 5

        lidar_image = np.zeros(
            (bev_size, bev_size, 3),
            dtype=np.uint8
        )

        x = (xy[:, 0] * scale + bev_size / 2).astype(np.int32)
        y = (-xy[:, 1] * scale + bev_size / 2).astype(np.int32)

        mask = (
            (x >= 0) &
            (x < bev_size) &
            (y >= 0) &
            (y < bev_size)
        )

        lidar_image[y[mask], x[mask]] = 255

        cv2.circle(
            lidar_image,
            (bev_size // 2, bev_size // 2),
            5,
            (0, 0, 255),
            -1
        )

        lidar_image = cv2.resize(
            lidar_image,
            (400, 450)
        )

        dashboard = np.hstack(
            (
                rgb_image,
                lidar_image
            )
        )

        velocity = vehicle.get_velocity()
        speed = velocity.length() * 3.6

        cv2.putText(
            dashboard,
            f"Speed: {speed:.1f} km/h",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            dashboard,
            f"Frame: {image.frame}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
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

        cv2.imshow(
            "CARLA Sensor Dashboard",
            dashboard
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    camera.stop()
    lidar.stop()

    camera.destroy()
    lidar.destroy()
    vehicle.destroy()

    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(original_settings)

    cv2.destroyAllWindows()

    print("Actors destroyed")
    print("World settings restored")