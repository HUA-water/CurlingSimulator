import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


memory = np.load('collision_memory.npy', allow_pickle=True)
_memory = []
for data in memory:
	flag = False
	eps = 0.1
	if (np.linalg.norm(data[2]) < 0.01):
		if np.abs(data[1][0]) > eps:
			flag = True
		if data[1][1] - data[0][1] > eps:
			flag = True
		if data[3][1] > eps:
			flag = True
		if np.abs(data[3][0] - data[0][0]) > eps:
			flag = True
	if flag:
		_memory.append(data)
	continue
	print(data)
	plt.plot([0.29, 0.29-data[0][0]], [0, -data[0][1]], c='red')
	plt.plot([0.29, 0.29+data[1][0]], [0, data[1][1]], c='#FF1493')
	plt.plot([0, data[2][0]], [0, data[2][1]], c='#4B0082')
	plt.plot([0, data[3][0]], [0, data[3][1]], c='blue')
	plt.xlim(-1, 1)
	plt.ylim(-1, 1)
	plt.show()
print(len(memory), len(_memory))
memory = np.array(_memory)
inputs = np.concatenate([memory[:, 0], memory[:, 2]], axis = 1)
outputs = np.concatenate([memory[:, 1], memory[:, 3]], axis = 1)
	
def getCollisionModel():
	x = tf.keras.Input(4)
	m = tf.keras.layers.Dense(8, activation = 'relu', kernel_initializer = 'random_uniform')(x)
	y = tf.keras.layers.Dense(4, kernel_initializer = 'random_uniform')(m)
	model = tf.keras.Model(inputs = x, outputs = y)
	model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-8), loss=["mse"], metrics = ["mae"])
	return model
collision_model = getCollisionModel()
'''
collision_model = tf.keras.models.load_model("collision.hdf5")
collision_model.summary()

pred = collision_model.predict(inputs)
for i in range(10):
	print(outputs[i])
	print(pred[i])
	print()
exit(0)
'''
collision_model_checkpoint = tf.keras.callbacks.ModelCheckpoint("collision.hdf5", monitor='loss', verbose=0, save_best_only=True)
log_dir="logs/collision"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
try:
	collision_model.fit(inputs, outputs, validation_split = 0.2, epochs=1000, callbacks=[collision_model_checkpoint, tensorboard_callback])
except:
	pass

for i in range(len(collision_model.get_layer('dense').get_weights())):
	print(collision_model.get_layer('dense').get_weights()[i])
	print('******************************************************')
for i in range(len(collision_model.get_layer('dense_1').get_weights())):
	print(collision_model.get_layer('dense').get_weights()[i])
	print('******************************************************')