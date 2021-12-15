import numpy
# from colors import Color
red = blue = lambda x: str(x)

class ConnectFour():
  ROWS = 6
  COLS = 7

  def __init__(self):
    self.board = numpy.zeros((self.COLS, self.ROWS), dtype = 'i')
    self.heights = numpy.zeros(self.COLS, dtype = 'i')
    self.turn = 1 
    self.history = []

  def on_board(self, x, y):
    return x >= 0 and x < self.COLS and y >= 0 and y < self.ROWS

  def scan(self, x, y, dx, dy):
    c = 0
    p = self.board[x, y]
    while self.on_board(x, y) and self.board[x, y] == p:
      c+=1
      x+=dx
      y+=dy
    return c
 
  def check_win(self, x):
    y = self.heights[x] - 1
    if self.scan(x, y, 0, -1) >= 4: return True
    if self.scan(x, y, 1, 1) + self.scan(x, y, -1,-1) - 1 >= 4: return True
    if self.scan(x, y, 1,-1) + self.scan(x, y, -1, 1) - 1 >= 4: return True
    if self.scan(x, y, 1, 0) + self.scan(x, y, -1, 0) - 1 >= 4: return True
    return False
      
  def make_move(self, x):
    self.board[x, self.heights[x]] = self.turn
    self.heights[x]+=1
    self.turn*=-1
    self.history.append(x)
    return self.check_win(x)

  def unmake_move(self):
    x = self.history.pop()
    self.heights[x]-=1
    self.board[x,self.heights[x]] = 0
    self.turn*=-1

  def moves(self):
    return [x for x in range(self.COLS) if self.heights[x] < self.ROWS]

  # print the board
  def show(self):
    print("Player one: ●           Player two: △")
    for y in range(self.ROWS):
      print('|', end = '')
      for x in range(self.COLS):
        if self.board[x, self.ROWS - y - 1] == 1:
          print(red('●') + '|', end = '')
        elif self.board[x, self.ROWS - y -  1] == -1:
          print(blue('△') + '|', end = '')
        else:
          print(' |', end = '')
      print('')
    print('+-+-+-+-+-+-+-+')
    if len(self.history) > 0:
      print(' ', end = '')
      last_move = self.history[-1]
      for x in range(self.COLS):
        if last_move == x:
          print('^', end = '')
        else:
          print('  ', end = '')
      print('')
