import cv2
import math
import tensorflow as tf
import numpy as np
import time


Balls = []
START_POINT = [2.375, 27.6+4.88]
MAX_POINT = [4.75, 13]
AIR_DRAG = [0.03319, 0.0331, 0.033] #空气阻力，与速度有关
FRICTION = [0.05, 0.05, 0.0673] #摩擦力，与速度无关
MIN_VELOCITY = 1e-3
MIN_ANGLE = 1e-2
ANGLE_LOSS = [0.24, 0.22, 0.22] #转角损耗
VELOCITY_LOSS_ANGLE = [0.00072, 0, 0] #转角的存在导致的速度损耗
VELOCITY_ANGLE = [0.002, 0.002, 0.002] #转角的存在导致的速度方向改变
BALL_R = 0.145 #冰壶半径
DELTA_TIEM = 0.01 #离散时间间隔
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
					print(i, 'C:', self.Balls[i].coordinate - 4.88j - 2.375, 'V:', self.Balls[i].velocity, 'A:', self.Balls[i].angle)
			'''
			round += 1
			while True:
				flag_COLL = 1
				for i in range(N):
					for j in range(i+1, N):
						if collision_record[i][j] < round - 100 and distance(self.Balls[i].coordinate + self.Balls[i].velocity * DELTA_TIEM, self.Balls[j].coordinate + self.Balls[j].velocity * DELTA_TIEM) <= 2*BALL_R:
							delta_t = DELTA_TIEM
							for times in range(10):
								delta_t/=2
								A = self.Balls[i].coordinate + self.Balls[i].velocity * delta_t
								B = self.Balls[j].coordinate + self.Balls[j].velocity * delta_t
								if distance(A, B) >= 2*BALL_R:
									self.Balls[i].coordinate = A
									self.Balls[j].coordinate = B
									
							flag_COLL = 1
							collision_record[i][j] = round
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
							deltaV = self.Balls[i].velocity - self.Balls[j].velocity
							F = (deltaC.conjugate()*deltaV)
							print(self.Balls[i].velocity, self.Balls[j].velocity, deltaC, F)
							if (np.abs(F.real) > 2):
								F = F.real
							else:
								F = F.real + F.imag * 2j# * 0.5j
								F *= COLLISION
							self.Balls[i].velocity -= F*deltaC
							self.Balls[j].velocity += F*deltaC
							self.Balls[i].velocity *= 1 - COLLISION_LOSS
							self.Balls[j].velocity *= 1 - COLLISION_LOSS
							'''
				if flag_COLL:
					break
			
			flag = 1
			move = 0
			for ball in self.Balls:
				ball.coordinate += ball.velocity * DELTA_TIEM
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
					ball.velocity -= ball.velocity / Abs * DELTA_TIEM * FRICTION[stage]
					ball.velocity -= ball.velocity * Abs * AIR_DRAG[stage] * DELTA_TIEM
					
					ball.velocity *= (1 - VELOCITY_LOSS_ANGLE[stage]*np.abs(ball.angle))**DELTA_TIEM
					Rot = (1 + 1j * ball.angle * VELOCITY_ANGLE[stage]) ** DELTA_TIEM
					ball.velocity *= Rot/np.abs(Rot)
					ball.angle *= (1 - ANGLE_LOSS[stage])**DELTA_TIEM
				else:
					ball.velocity = 0
					ball.angle = 0
				if np.abs(ball.coordinate.imag-21.52)<0.01:
					print(ball.coordinate, ball.angle, ball.velocity, np.abs(ball.velocity))
			if ((int(timeNow/DELTA_TIEM)&31) == 0 and move == 1):
				self.draw()
			timeNow += DELTA_TIEM
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
	