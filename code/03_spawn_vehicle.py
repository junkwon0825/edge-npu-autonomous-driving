import carla
import random
import time

#1. data
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
traffic_manager = client.get_trafficmanager(8000)

original_settings = world.get_settings()

#2. model
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager.set_synchronous_mode(True)

blueprints = world.get_blueprint_library()
vehicle_bp = blueprints.filter("vehicle.tesla.model3")[0]
spawn_points = world.get_map().get_spawn_points()

#3. compile,train
vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
vehicle.set_autopilot(True, traffic_manager.get_port())

spectator = world.get_spectator()

#4. evaluate,predict
print("Vehicle ID:", vehicle.id)
print("Synchronous mode ON")

try:
    while True:
        world.tick()

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

except KeyboardInterrupt:
    print("Stopping...")

finally:
    vehicle.destroy()

    traffic_manager.set_synchronous_mode(False)

    world.apply_settings(original_settings)

    print("Vehicle destroyed")
    print("World settings restored")