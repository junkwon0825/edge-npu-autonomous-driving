import carla

#1. data
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

#2. model
world = client.get_world()
map_name = world.get_map().name

#3. compile,train
actors = world.get_actors()

#4. evaluate,predict
print("Map:", map_name)
print("Actors:", len(actors))