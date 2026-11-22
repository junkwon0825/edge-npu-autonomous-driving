import carla
import random
import time
import os

#1. data
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

blueprints = world.get_blueprint_library()
vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
camera_bp = blueprints.find("sensor.camera.rgb")

camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "450")
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", "0.1")

spawn_points = world.get_map().get_spawn_points()

save_dir = r"<LOCAL_PROJECT_ROOT>\data\rgb"
os.makedirs(save_dir, exist_ok=True)

#2. model
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
vehicle.set_autopilot(True)

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=1.7),
    carla.Rotation(pitch=0)
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

spectator = world.get_spectator()

#3. compile,train
def save_image(image):
    file_path = os.path.join(save_dir, f"{image.frame:06d}.png")
    image.save_to_disk(file_path)
    print(f"Saved: {file_path}")

camera.listen(save_image)

#4. evaluate,predict
print("Vehicle ID:", vehicle.id)
print("Camera ID:", camera.id)
print("Camera listening:", camera.is_listening)
print("Save directory:", save_dir)

try:
    while True:
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

        velocity = vehicle.get_velocity()
        speed = velocity.length() * 3.6

        print(f"Speed: {speed:.2f} km/h")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    print("Actors destroyed")