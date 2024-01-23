import torch
import numpy as np
import pygame
import random
print("Finished importing")

class Game2048():
    new_tile_distribution = [1]*9+[2]
    def __init__(self, board=None, score=None) -> None:
        if board is None:
            self.board = torch.zeros((4, 4), dtype=torch.uint8)
            self.zeros=16
        else:
            self.board = board
            self.zeros=0
            for i,j in [(i,j) for i in range(4) for j in range(4)]:
                if self.board[i,j] == 0:
                    self.zeros += 1
        if score is None:
            self.score = 0
        else:
            self.score = score
        self.done = False
        self.add_tile()
        self.add_tile()
    
    def add_tile(self):
        if self.zeros == 0:
            raise(Exception('Tried to add tile to full board, should not happen'))
        rand_index=random.randint(1,self.zeros)
        counter = 1
        for i in range(4):
            for j in range(4):
                if self.board[i, j] == 0:
                    if counter == rand_index:
                        self.board[i, j] = random.choice(Game2048.new_tile_distribution)
                        # print("added zero at", i, j, "because rand_index was", rand_index, "and counter was", counter)
                        self.zeros -= 1
                        return
                    counter+=1
        raise(Exception("Something went wrong in add_tile, should have returned by now"))
        
    def move(self, direction):
        if direction == "left":
            something_changed = self.move_left()
        elif direction == "up":
            something_changed = self.move_up() 
        elif direction == "right":
            something_changed = self.move_right()
        elif direction == "down":
            something_changed = self.move_down()
        else:
            raise(Exception("Invalid direction"))

        if self.zeros==0 and not something_changed:
            for i in range(4):
                for j in range(3):
                    if self.board[i,j]==self.board[i,j+1] or self.board[j,i]==self.board[j+1,i]:
                        return False
            return True
        return False

    def move_left(self):
        something_changed = False
        for row_no in range(4):
            left_most_zero = -1
            closest_num = (-1,-1)
            for i, num in enumerate(self.board[row_no]):
                if num == 0 and left_most_zero == -1:    
                    left_most_zero = i    
                if closest_num[1]==num:    
                    self.score += (2**(num+1)).item()
                    self.board[row_no, closest_num[0]] = num+1
                    self.board[row_no, i] = 0
                    left_most_zero = closest_num[0]+1
                    closest_num = (-1,-1)
                    self.zeros+=1
                    something_changed = True    
                    continue
                if num != 0 and left_most_zero != -1:    
                    closest_num = (left_most_zero,num.item())    
                    self.board[row_no, left_most_zero] = num
                    self.board[row_no, i] = 0
                    something_changed = True
                    left_most_zero += 1    
                    continue
                if num != 0:    
                    closest_num = (i, num)    
                    continue
        if self.zeros!=0 and something_changed:
            self.add_tile()
        return something_changed

    def move_right(self):
        something_changed = False
        for row_no in range(4):
            right_most_zero = -1
            closest_num = (-1,-1)
            for i, num in enumerate(reversed(self.board[row_no])):
                i=3-i
                if num == 0 and right_most_zero == -1:    
                    right_most_zero = i    
                if closest_num[1]==num:    
                    self.score += (2**(num+1)).item()
                    self.board[row_no, closest_num[0]] = num+1
                    self.board[row_no, i] = 0
                    right_most_zero = closest_num[0]-1
                    closest_num = (-1,-1)
                    self.zeros+=1
                    something_changed = True    
                    continue
                if num != 0 and right_most_zero != -1:    
                    closest_num = (right_most_zero,num.item())    
                    self.board[row_no, right_most_zero] = num
                    self.board[row_no, i] = 0
                    right_most_zero -= 1    
                    something_changed = True
                    continue
                if num != 0:    
                    closest_num = (i, num)    
                    continue
        if self.zeros!=0 and something_changed:
            self.add_tile()
        return something_changed

    def move_up(self):
        something_changed = False
        for col_no in range(4):
            upper_most_zero = -1
            closest_num = (-1,-1)
            for i, num in enumerate(self.board[:,col_no]):
                if num == 0 and upper_most_zero == -1:    
                    upper_most_zero = i    
                if closest_num[1]==num:    
                    self.score += (2**(num+1)).item()
                    self.board[closest_num[0], col_no] = num+1
                    self.board[i, col_no] = 0
                    upper_most_zero = closest_num[0]+1
                    closest_num = (-1,-1)
                    self.zeros+=1
                    something_changed = True    
                    continue
                if num != 0 and upper_most_zero != -1:    
                    closest_num = (upper_most_zero,num.item())    
                    self.board[upper_most_zero, col_no] = num
                    self.board[i, col_no] = 0
                    upper_most_zero += 1    
                    something_changed = True
                    continue
                if num != 0:    
                    closest_num = (i, num)    
                    continue
        if self.zeros!=0 and something_changed:
            self.add_tile()
        return something_changed

    def move_down(self):
        something_changed = False
        for col_no in range(4):
            upper_most_zero = -1
            closest_num = (-1,-1)
            for i, num in enumerate(reversed(self.board[:,col_no])):
                i=3-i
                if num == 0 and upper_most_zero == -1:    
                    upper_most_zero = i    
                if closest_num[1]==num:    
                    self.score += (2**(num+1)).item()
                    self.board[closest_num[0], col_no] = num+1
                    self.board[i, col_no] = 0
                    upper_most_zero = closest_num[0]-1
                    closest_num = (-1,-1)
                    self.zeros+=1
                    something_changed = True    
                    continue
                if num != 0 and upper_most_zero != -1:    
                    closest_num = (upper_most_zero,num.item())    
                    self.board[upper_most_zero, col_no] = num
                    self.board[i, col_no] = 0
                    upper_most_zero -= 1    
                    something_changed = True
                    continue
                if num != 0:    
                    closest_num = (i, num)    
                    continue
        if self.zeros!=0 and something_changed:
            self.add_tile()
        return something_changed

    rendering = False
    def render(self):
        if not self.rendering:
            self.init_render()
            
        # Limit to 30 fps
        self.clock.tick(30)
     
        # Clear the screen
        self.screen.fill((187,173,160))
        
        # Draw board
        colors = [(205,193,180), (238,228,218), (237,224,200), (242,177,121), 
                  (245,149,99), (246,124,95), (246,94,69), (237,204,121), 
                  (237,204,97), (237,197,63), (121,204,237), (97,177,237), 
                  (63,149,204), (121,121,177), (40,40,80), (20,20,60)]
        
        border = 10
        pygame.draw.rect(self.screen, (187,173,160), pygame.Rect(100,0,600,600))
        for i in range(4):
            for j in range(4):
                exponentiated = torch.zeros((4,4), dtype=torch.int32)
                exponentiated.copy_(self.board)
                exponentiated = 2**exponentiated
                exponentiated[exponentiated==1] = 0
                val = exponentiated[i][j]
                validx = int(np.log2(val)) if val>0 else 0
                pygame.draw.rect(self.screen, colors[validx % len(colors)], pygame.Rect(100+150*j+border,150*i+border,150-2*border,150-2*border))
                if val>0:
                    text = self.font.render("{:}".format(val), True, (255,255,255))                
                    x = 175 + 150*j - text.get_width()/2
                    y = 75 + 150*i - text.get_height()/2                
                    self.screen.blit(text, (x, y))
        text = self.scorefont.render("{:}".format(self.score), True, (0,0,0))
        self.screen.blit(text, (790-text.get_width(), 10))

        # Display
        pygame.display.flip()

    def init_render(self):
        self.screen = pygame.display.set_mode([800, 600])
        pygame.display.set_caption('2048')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

        # Set up game
        self.font = pygame.font.Font(None, 50)
        self.scorefont = pygame.font.Font(None, 30)

    def __str__(self):
        exponentiated = torch.zeros((4,4), dtype=torch.int32)
        exponentiated.copy_(self.board)
        exponentiated = 2**exponentiated
        exponentiated[exponentiated==1] = 0
        return "Board: "+str(self.board)[7:]+"\nScore: "+str(self.score)

def play():
    env = Game2048()
    actions = ['left', 'right', 'up', 'down']
    exit_program = False
    while not exit_program:
        env.render()

        # Process game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    exit_program = True
                if event.key == pygame.K_UP:
                    exit_program = env.move("up")
                if event.key == pygame.K_DOWN:
                    exit_program = env.move("down")
                if event.key == pygame.K_RIGHT:
                    exit_program = env.move("right")
                if event.key == pygame.K_LEFT:
                    exit_program = env.move("left")
    print("ded")

if __name__ == "__main__":
    pygame.init()
    play()
    exit_program = False
    while not exit_program:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_program = True
            if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        play()
    pygame.quit()
