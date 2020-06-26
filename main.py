import cv2
import math
import numpy as np
import time


Balls = []
START_POINT = [2.35, 27.6+4.88]
MAX_POINT = [4.75, 13]
AIR_DRAG = 0.1025 #空气阻力，与速度有关
FRICTION = 0 #摩擦力，与速度无关
MIN_VELOCITY = 1e-2
MIN_ANGLE = 1e-2
ANGLE_LOSS = 0.21 #转角损耗
VELOCITY_LOSS_ANGLE = 0.00059 #转角的存在导致的速度损耗
VELOCITY_ANGLE = 0.001833 #转角的存在导致的速度方向改变
BALL_R = 0.24 #冰壶半径
DELTA_TIEM = 0.01 #离散时间间隔
COLLISION = 0.6 #碰撞能量损耗


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
		
	def addBall(self, ball):
		self.Balls.append(ball)
	
	def draw(self):
		#return
		img = np.zeros((int(MAX_POINT[0]*100), int(MAX_POINT[1]*100), 3))
		for ball in self.Balls:
			if inImg(ball.coordinate):
				x = ball.coordinate.real
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
		time = 0
		N = len(self.Balls)
		while True:
			flag = 1
			move = 0
			while True:
				flag = 1
				for i in range(N):
					for j in range(i+1, N):
						if distance(self.Balls[i].coordinate + self.Balls[i].velocity * DELTA_TIEM, self.Balls[j].coordinate + self.Balls[j].velocity * DELTA_TIEM) < 2*BALL_R:
							flag = 0
							deltaC = self.Balls[i].coordinate - self.Balls[j].coordinate
							deltaC /= np.abs(deltaC)
							deltaV = self.Balls[j].velocity - self.Balls[i].velocity
							F = (deltaC*deltaV.conjugate()).real
							F *= COLLISION
							self.Balls[i].velocity += F*deltaC
							self.Balls[j].velocity -= F*deltaC
				if flag:
					break
			
			dv = DELTA_TIEM * FRICTION
			for ball in self.Balls:
				ball.coordinate += ball.velocity * DELTA_TIEM
				Abs = np.abs(ball.velocity)
				if Abs>MIN_VELOCITY or ball.angle > MIN_ANGLE:
					if inImg(ball.coordinate):
						move = 1
					flag = 0
					ball.velocity -= ball.velocity/Abs*dv
					ball.velocity *= (1 - AIR_DRAG)**DELTA_TIEM
					
					ball.velocity *= (1 - VELOCITY_LOSS_ANGLE*np.abs(ball.angle))**DELTA_TIEM
					Rot = (1 + 1j * ball.angle * VELOCITY_ANGLE) ** DELTA_TIEM
					ball.velocity *= Rot/np.abs(Rot)
					ball.angle *= (1 - ANGLE_LOSS)**DELTA_TIEM
				else:
					ball.velocity = 0
					ball.angle = 0
				if np.abs(ball.coordinate.imag-21.51)<0.01:
					print(ball.coordinate, ball.angle, np.abs(ball.velocity))
			if ((int(time/DELTA_TIEM)&31) == 0 and move == 1):
				self.draw()
			time += DELTA_TIEM
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
	