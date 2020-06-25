import cv2
import math
import numpy as np
import time


Balls = []
START_POINT = [2.325, 27.6+4.88]
MAX_POINT = [4.75, 13]
FRICTION = 0.1623 #摩擦力
BALL_R = 0.24 #冰壶半径
DELTA_TIEM = 0.01 #离散时间间隔
COLLISION = 1 #碰撞能量损耗


def inImg(P):
	x = P.real
	y = P.imag
	return x>=0 and y>=0 and x<MAX_POINT[0] and y<MAX_POINT[1]
def distance(X, Y, L = 2):
	return ((X.real-Y.real)**L+(X.imag-Y.imag)**L)**(1/L)
    
class Ball:
	def __init__(self, vx, vy):
		self.coordinate = START_POINT[0] + 1j*START_POINT[1]
		self.velocity = vx + 1j*vy;

class Curling:
	Balls = []
	def addBall(self, ball):
		self.Balls.append(ball)
	
	def draw(self):
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
			dv = DELTA_TIEM * FRICTION
			for i in range(N):
				for j in range(i+1, N):
					if distance(self.Balls[i].coordinate, self.Balls[j].coordinate) < 2*BALL_R:
						deltaC = self.Balls[i].coordinate - self.Balls[j].coordinate
						deltaV = self.Balls[i].velocity - self.Balls[j].velocity
						F = (deltaC*deltaV.conjugate())/(deltaC*deltaC.conjugate())
						F *= COLLISION
						self.Balls[i].velocity -= F*deltaC
						self.Balls[j].velocity += F*deltaC
						
			
			for ball in self.Balls:
				ball.coordinate += ball.velocity * DELTA_TIEM
				Abs = np.abs(ball.velocity)
				if Abs>dv:
					if inImg(ball.coordinate):
						move = 1
					flag = 0
					ball.velocity -= ball.velocity/Abs*dv
				else:
					ball.velocity = 0
			if ((int(time/DELTA_TIEM)&3) == 0 and move == 1):
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
		vy, vx = str.split(' ')
		vx = float(vx)
		vy = float(vy)
		
		Platform.addBall(Ball(vx, -vy))
		Platform.run()
	