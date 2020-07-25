import cv2
import math
import tensorflow as tf
import numpy as np
import time
import math


Balls = []
START_POINT = [2.35, 27.6+4.88]
MAX_POINT = [4.75, 13]
AIR_DRAG = [0.0326, 0.0329, 0.03] #空气阻力，与速度有关
FRICTION = [0.05, 0.05, 0.0689] #摩擦力，与速度无关
MIN_VELOCITY = 1e-2
MIN_ANGLE = 1000
ANGLE_LOSS = [0.2127, 0.214, 0.2127] #转角损耗
VELOCITY_LOSS_ANGLE = [0.00065, 0, 0] #转角的存在导致的速度损耗
VELOCITY_ANGLE = [0.00188, 0.0, 0.00182] #转角的存在导致的速度方向改变
ANGLE_INCRESS_VELOCITY = [0, 0, 2.25] #最后一个阶段会有速度导致转角出现的情况
BALL_R = 0.145 #冰壶半径
DELTA_TIME = 0.2 #离散时间间隔
COLLISION = 0.5 #碰撞力的损耗
COLLISION_LOSS = 0 #碰撞产生速度削减
DRAW = 1


def inImg(P):
	x = P.real
	y = P.imag
	return x>=0 and y>=0 and x<MAX_POINT[0] and y<MAX_POINT[1]
def distance(X, Y, L = 2):
	return ((X.real-Y.real)**L+(X.imag-Y.imag)**L)**(1/L)
    
class Ball:
	def __init__(self, vy, dx, angle):
		self.coordinate = START_POINT[0] + dx + 1j*START_POINT[1]
		self.velocity = 1j*vy;
		self.angle = angle

class Curling:
	Balls = []
	def __init__(self):
		self.draw()
		self.collision_model = tf.keras.models.load_model("collision.hdf5")
		
	def addBall(self, ball):
		self.Balls.append(ball)
	
	def draw(self):
		if DRAW == False:
			return
		img = np.zeros((int(MAX_POINT[0]*100), int(MAX_POINT[1]*100), 3))
		for ball in self.Balls:
			if inImg(ball.coordinate):
				x = MAX_POINT[0] - ball.coordinate.real
				y = ball.coordinate.imag
				color = [255, 255, 255]
				CX = int(x*100)
				CY = int(y*100)
				R = int(BALL_R*100)
				for X in range(CX-R, CX+R+1):
					if X >= img.shape[0] or X < 0:
						continue
					_R = int((R**2-(X - CX)**2)**0.5)
					for Y in range(CY-_R, CY+_R+1):
						if Y >= img.shape[1] or Y < 0:
							continue
						img[X][Y] = color
		cv2.imshow('Curling', img)
		cv2.waitKey(1)
		
	def run(self):
		timeNow = 0
		N = len(self.Balls)
		round = 0
		collision_record = np.zeros((N, N))
		while True:
			'''
			if timeNow <= 100:
				print(timeNow)
				for i in range(N):
					print(i, 'C:', self.Balls[i].coordinate, 'V:', self.Balls[i].velocity, math.atan2(self.Balls[i].velocity.imag, self.Balls[i].velocity.real), np.abs(self.Balls[i].velocity), 'A:', self.Balls[i].angle)
			'''
			round += 1
			delta_time = DELTA_TIME
			while True:
				flag_COLL = 1
				for i in range(N):
					for j in range(i+1, N):
						if collision_record[i][j] < round - 50:
							deltaC = self.Balls[i].coordinate - self.Balls[j].coordinate
							deltaV = self.Balls[i].velocity - self.Balls[j].velocity
							tmp = -deltaC*deltaV.conjugate() / np.abs(deltaV)
							if (np.abs(tmp.imag) < 2 * BALL_R and tmp.real > 0):
								time_cost = (tmp.real - ((2*BALL_R)**2 - tmp.imag**2)**0.5)/np.abs(deltaV)
								#print(tmp, deltaV, time_cost)
								if time_cost > 0.001:
									time_cost -= 1e-5
									if (delta_time > time_cost):
										delta_time = time_cost
								else:
									flag_COLL = 0
									collision_record[i][j] = round
									'''
									x, y = i, j
									if np.abs(self.Balls[x].velocity) < np.abs(self.Balls[y].velocity):
										x, y = y, x
									deltaC = self.Balls[x].coordinate - self.Balls[y].coordinate
									deltaC /= np.abs(deltaC)
									print(x, y, self.Balls[x].velocity, self.Balls[y].velocity)
									Vx = self.Balls[x].velocity * deltaC.conjugate()
									Vy = self.Balls[y].velocity * deltaC.conjugate()
									
									time_cost = time.time()
									input = [[Vx.real, Vx.imag, Vy.real, Vy.imag]]
									output = self.collision_model.predict(input)[0]
									time_cost = time.time() - time_cost
									
									self.Balls[x].velocity = (output[0] + 1j*output[1]) * deltaC
									self.Balls[y].velocity = (output[2] + 1j*output[3]) * deltaC
									print(x, y, self.Balls[x].velocity, self.Balls[y].velocity)
									print(time_cost)
									'''
									deltaC = self.Balls[i].coordinate - self.Balls[j].coordinate
									deltaC /= np.abs(deltaC)
									deltaV = self.Balls[i].velocity - self.Balls[j].velocity
									F = (deltaC.conjugate()*deltaV)
									angleCos = np.abs(F.real)/np.abs(deltaV)
									#print(self.Balls[i].velocity, self.Balls[j].velocity, deltaC, F)
									if (np.abs(F.real) > 2):
										F = F.real
									else:
										print('==========', angleCos)
										F = F.real + F.imag * (angleCos) * 1j
										F *= COLLISION
									self.Balls[i].velocity -= F*deltaC
									self.Balls[j].velocity += F*deltaC
									self.Balls[i].velocity *= 1 - COLLISION_LOSS
									self.Balls[j].velocity *= 1 - COLLISION_LOSS
									
				if flag_COLL:
					break
			
			flag = 1
			move = 0
			for ball in self.Balls:
				velocity_ = ball.velocity
				Abs = np.abs(ball.velocity)
				if Abs > 1.5:
					stage = 0
				elif Abs > 1:
					stage = 1
				else:
					stage = 2
					
				if Abs>MIN_VELOCITY or ball.angle > MIN_ANGLE:
					if inImg(ball.coordinate):
						move = 1
					flag = 0
					if Abs < delta_time * FRICTION[stage]:
						ball.velocity = 0
					else:
						ball.velocity -= ball.velocity / Abs * delta_time * FRICTION[stage]
						ball.velocity -= ball.velocity * Abs * AIR_DRAG[stage] * delta_time
					
					ball.velocity *= (1 - VELOCITY_LOSS_ANGLE[stage]*np.abs(ball.angle))**delta_time
					Rot = (1 + 1j * ball.angle * VELOCITY_ANGLE[stage]) ** delta_time
					ball.velocity *= Rot/np.abs(Rot)
					ball.angle *= (1 - ANGLE_LOSS[stage])**delta_time
				else:
					ball.velocity = 0
					ball.angle = 0
				
				ball.angle += (1 - np.abs(ball.velocity)) * ANGLE_INCRESS_VELOCITY[stage] * delta_time
					
				ball.coordinate += (ball.velocity + velocity_) / 2 * delta_time
				
				if np.abs(ball.coordinate.imag-21.52)<0.01:
					print(ball.coordinate, ball.angle, ball.velocity, np.abs(ball.velocity))
			if ((int(timeNow/delta_time)&31) == 0 and move == 1):
				self.draw()
			timeNow += delta_time
			if flag:
				break
		self.draw()
		for ball in self.Balls:
			print(ball.coordinate.real, ball.coordinate.imag)
		

if __name__ == "__main__":
	Platform = Curling()
	while True:
		str = input()
		if (str[0] == 'E'):
			break
		vy, dx, angle = str.split(' ')
		vy = float(vy)
		dx = float(dx)
		angle = float(angle)
		
		Platform.addBall(Ball(-vy, dx, angle))
		Platform.run()
	