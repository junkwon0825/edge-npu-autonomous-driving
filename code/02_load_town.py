import carla

#1. data
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)

#2. model
world = client.load_world("Town01")

#3. compile,train
map_name = world.get_map().name

#4. evaluate,predict
print("Loaded map:", map_name)