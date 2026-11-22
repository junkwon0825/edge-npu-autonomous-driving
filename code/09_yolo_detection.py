import carla
import random
import numpy as np
import cv2
import torch

from queue import Queue, Empty
from ultralytics import YOLO

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

camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "450")
camera_bp.set_attribute("fov", "90")
camera_bp.set_attribute("sensor_tick", "0.05")

spawn_points = world.get_map().get_spawn_points()

#2. model
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolo11n.pt")

vehicle = world.spawn_actor(
    vehicle_bp,
    random.choice(spawn_points)
)

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)

camera = world.spawn_actor(
    camera_bp,
    carla.Transform(
        carla.Location(x=1.5, z=1.7)
    ),
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid
)

camera_queue = Queue()
spectator = world.get_spectator()

#3. compile,train
camera.listen(camera_queue.put)

print("Device:", device)
print("YOLO Detection ON")

camera.listen(camera_queue.put)

print("Device:", device)
print("YOLO Detection ON")

for _ in range(10):
    world.tick()

print("Camera warm-up complete")

#4. evaluate,predict
try:
    while True:
        world.tick()

        try:
            image = camera_queue.get(timeout=5.0)

        except Empty:
            print("Camera timeout")
            print("Camera alive:", camera.is_alive)
            print("Camera listening:", camera.is_listening)
            continue
        image_array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )

        image_array = np.reshape(
            image_array,
            (image.height, image.width, 4)
        )

        frame = image_array[:, :, :3]

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.4,
            device=device,
            verbose=False
        )

        detected_frame = results[0].plot()

        velocity = vehicle.get_velocity()
        speed = velocity.length() * 3.6

        cv2.putText(
            detected_frame,
            f"Speed: {speed:.1f} km/h",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
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
            "CARLA YOLO Detection",
            detected_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    camera.stop()

    camera.destroy()
    vehicle.destroy()

    traffic_manager.set_synchronous_mode(False)
    world.apply_settings(original_settings)

    cv2.destroyAllWindows()

    print("Actors destroyed")
    print("World settings restored")