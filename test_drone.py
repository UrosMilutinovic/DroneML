import tensorflow as tf
from codrone_edu.drone import *

drone = Drone()
drone.pair()
print("Paired!")
drone.takeoff()
print("In the air!")
drone.hover()
direction = input("Input a command: ")


if direction == "e":
    print("Landing")
    drone.land()
    drone.close()
    print("Program complete")
elif direction == "w":
    print("forward")
    drone.land()
    drone.close()
    

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)