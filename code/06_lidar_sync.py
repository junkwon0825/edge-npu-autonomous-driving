import carla
import random
import os
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

rgb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rgb")
lidar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lidar")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(lidar_dir, exist_ok=True)

#2. model
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
vehicle.set_autopilot(True, traffic_manager.get_port())

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=1.7)
)

lidar_transform = carla.Transform(
    carla.Location(x=0.0, z=2.2)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

lidar = world.spawn_actor(
    lidar_bp,
    lidar_transform,
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

spectator = world.get_spectator()

camera_queue = Queue()
lidar_queue = Queue()

#3. compile,train
camera.listen(camera_queue.put)
lidar.listen(lidar_queue.put)

#4. evaluate,predict
print("Vehicle ID:", vehicle.id)
print("Camera ID:", camera.id)
print("LiDAR ID:", lidar.id)
print("Camera + LiDAR synchronous mode ON")

frame_count = 0

try:
    while True:
        world.tick()

        try:
            image = camera_queue.get(timeout=2.0)
            lidar_data = lidar_queue.get(timeout=2.0)

        except Empty:
            print("Sensor timeout")
            continue

        frame_count += 1

        print(
            f"World frame: {image.frame} | "
            f"Camera frame: {image.frame} | "
            f"LiDAR frame: {lidar_data.frame} | "
            f"LiDAR points: {len(lidar_data)}"
        )

        if image.frame != lidar_data.frame:
            print("WARNING: Sensor frames do not match")

        if frame_count % 100 == 0:
            image.save_to_disk(
                os.path.join(
                    rgb_dir,
                    f"{image.frame:06d}.png"
                )
            )

            lidar_data.save_to_disk(
                os.path.join(
                    lidar_dir,
                    f"{lidar_data.frame:06d}.ply"
                )
            )

            print("Sensor data saved")

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

    print("Actors destroyed")
    print("World settings restored")