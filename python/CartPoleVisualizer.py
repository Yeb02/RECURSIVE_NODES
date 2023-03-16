from CartPoleData import data

import pygame
from numpy import cos, sin


# Build and run the .exe with the CARTPOLE directive, and paste the console output into 
# CartPoleData.py's "data" list. Then, run this file to vizualize the results of the training !

#theta is exagerated for visualization
theta_factor = 10

pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Cartpole")
clock = pygame.time.Clock()
l = 100
ox, oy = 400, 300
cw, ch = 40, 20
f = 300

running = True       
for i in range(int(len(data)/2)):
    data[2*i+1] *= theta_factor
    screen.fill((0,0,0))

    pygame.draw.line(screen, (0, 0, 255), (0, oy), (600, oy))
    x = ox + f * data[2*i] + l * sin(data[2*i+1])
    y = oy - ch/2 -          l * cos(data[2*i+1])
    pygame.draw.line(screen, (0, 255, 0), (ox + f * data[2*i], oy - ch/2), (x,y))
    pygame.draw.rect(screen, (255, 0, 0), (ox + f * data[2*i]-cw/2, oy-ch/2, cw, ch))
    
    
    pygame.display.update()

    # fps = 1/tau, and standard tau is .02
    clock.tick(50)





