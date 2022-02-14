from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import time


def get_secant(x, y, p1, p2, srng):
    points = [p1, p2]
    m, b = np.polyfit(x[points], y[points], 1)
    return m * srng + b

def f(x): 
    return x**2

def derivative(x): 
    return 2*x

def tangent_line(x, x1, y1):
    return derivative(x1)*(x - x1) + y1

class Animation():

  def __init__(self, ax):
    self.ax = ax
    self.i = 0

  def secant_to_tangent(self, i):
    x = np.linspace(-5,5,200)
    # print(x[[10, -20]])

    # Choose point to plot tangent line
    p1 = 39
    p2 = 160 - self.i
    self.i += 1
    # print(self.i)



    y = f(x)
    x1 = x[39]
    y1 = y[39]

    self.ax.clear()
    self.ax.set_ylim([-2, np.amax(y)])

    trng = np.linspace(x1-1, x1+1, 10)
    tangent = tangent_line(trng, x1, y1)

    self.ax.plot(x, f(x), "r")
    scat1 = self.ax.scatter(x1, y1, color='g', s=50)
    self.ax.annotate("a", (x1, y1-2))
    line1, = self.ax.plot(trng, tangent, 'b', linewidth=2, label="tangent")
    line1.set_label("tangent")
    self.ax.set_ylabel("f(θ)")
    self.ax.set_xlabel("θ")


    if p1 < p2:
      srng = np.linspace(x[p1]-1, x[p2]+1, 50)
      secant = get_secant(x, y, p1, p2, srng)

      # self.ax.scatter(x1, y1, color='g', s=50)
      scat2 = self.ax.scatter(x[p2], f(x[p2]), color='g', s=50)
      self.ax.annotate("b", (x[p2], f(x[p2])-2))
      line2, = self.ax.plot(srng, secant, "g", linewidth=2, label="secant [a, b]")
      line2.set_label("secant [a, b]")

    self.ax.legend()


def true_func(slope, intercept):
  x = np.arange(1, 60)
  delta = np.random.uniform(-3,3, size=(x.size,))
  y = x*slope + intercept + delta
  return x, y

def prediction(x, w, b):
  return x*w + b

def loss_func(y_true, y_pred):
  return np.mean(np.square(y_true-y_pred), axis=-1)

def b_get_derivative(y, y_hat):
  return np.mean(-2*(y-y_hat))

def w_get_derivative(y, y_hat, x):
  return np.mean(-2*x*(y-y_hat))

def plot_derivative(d, loss, p, ax=None):
  x = np.linspace(p-1.5, p+1.5, 60)
  tangent = (x-p)*d + loss
  if ax is None:
    plt.plot(x, tangent, "b")
  else:
    ax.plot(x, tangent, "b")

class GradientDescent():

  def __init__(self, slope, intercept, si_loss=False):
    self.x, self.y = true_func(slope, intercept)
    self.true_w, self.true_b = slope, intercept
    self.w, self.b = [], []
    self.dw, self.db = [], []
    self.loss, self.y_hat = [], []
    self.si_loss = si_loss


  def loss_space(self):
    self.bias_linspace = np.linspace(min(np.amin(self.b), -3), max(11, np.amax(self.b)), 200)
    if not self.si_loss:
      self.intercept_losses = []
      for b in self.bias_linspace:
        # print(b)
        y_hat = prediction(self.x, self.true_w, b)
        self.intercept_losses.append(loss_func(self.y, prediction(self.x, self.true_w, b)))
    else:
      self.slope_linspace = np.linspace(min(-2.5, np.amin(self.w)), max(7, np.amax(self.w)), 200)
      W, B = np.meshgrid(self.slope_linspace, self.bias_linspace)
      pred = prediction(self.x[np.newaxis, ...], W[..., np.newaxis], B[..., np.newaxis])
      self.si_loss = loss_func(self.y, pred)
      self.slope_linspace = W
      self.bias_linspace = B


  def log(self, b, db, loss, y_hat, w=None, dw=None):
    self.b.append(b)
    self.db.append(db)
    self.loss.append(loss)
    self.y_hat.append(y_hat)
    if w is not None:
      self.w.append(w)
      self.dw.append(dw)

  def optimize_intercept(self):

    info = []

    lr = 0.01
    step_size = 1
    b = 9

    while abs(step_size) >= 0.001:

      y_hat = prediction(self.x, self.true_w, b)
      # Optional in this case because the loss
      # is implied in our derivative calculation,
      # but still a useful metric here
      loss = loss_func(self.y, y_hat)
      db = b_get_derivative(self.y, y_hat)

      self.log(b, db, loss, y_hat)

      step_size = db * lr
      b -= step_size

    return b


  def optimize_slope_intercept(self, epochs=1000):

    info = []

    wlr, blr = 1e-4, 0.1
    w_step_size = b_step_size = 1
    iters = 0
    b, w = 0.001, 7

    for i in range(epochs):

      iters += 1
      y_hat = prediction(self.x, w, b)
      # Optional in this case because the loss
      # is implied in our derivative calculation,
      # but still a useful metric here
      loss = loss_func(self.y, y_hat)
      db = b_get_derivative(self.y, y_hat)
      dw = w_get_derivative(self.y, y_hat, self.x)

      self.log(b, db, loss, y_hat, w, dw)

      b_step_size = db * blr
      w_step_size = dw * wlr
      b -= b_step_size
      w -= w_step_size

    return w, b

  def plot_residuals(self, w, b, ax=None, show=True, save=False):

    if ax is None:
      fig, ax = plt.subplots()

    y_hat = prediction(self.x, w, b)
    t, = ax.plot(self.x, self.y, "ro")
    f, = ax.plot(self.x, y_hat, "b")
    t.set_label("true values")
    f.set_label("fit")
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.vlines(self.x, self.y, y_hat)
    ax.set_title("Linear Regression")
    ax.legend()

    if show:
      plt.show()
    elif save:
      fig.savefig("/content/{}".format(save))

  def plot_derivative(self, d, loss, p, ax=None, show=True, label="b"):
    x = np.linspace(p-1.5, p+1.5, 60)
    tangent = (x-p)*d + loss
    if ax is None:
      plt.plot(x, tangent, "b")
    else:
      dv, = ax.plot(x, tangent, "b")
      dv.set_label("Derivative with respect to {}".format(label))

    if show:
      plt.show()


class OptimizationAnimation():

  def __init__(self, w, b, lim, func=None, obj=None):
    self.ax = None
    self.w = w
    self.b = b
    if func is not None:
      self.func = func
    self.lim = lim
    if obj is not None:
      self.obj = obj

  def linear_regression(self, i):
    self.ax.clear()
    self.ax.set_ylim(0, self.lim)
    self.func(self.w[i], self.b[i], ax=self.ax, show=False)

  def intercept_derivative(self, i):
    self.ax.clear()
    self.ax.set_ylim(0, self.lim)
    self.ax.set_xlim(np.amin(self.obj.bias_linspace), np.amax(self.obj.bias_linspace)+1)
    self.obj.plot_derivative(self.obj.db[i], self.obj.loss[i], self.b[i], ax=self.ax, show=False, label="intercept")
    ls, = self.ax.plot(self.obj.bias_linspace, self.obj.intercept_losses, "r")
    ls.set_label("Loss at intercept")
    self.ax.set_ylabel("Loss")
    self.ax.set_xlabel("Intercept")
    self.ax.legend()

  def gradient_descent(self, i):
    self.ax.clear()
    self.ax.set_zlim(0, np.amax(self.obj.si_loss)+1)
    self.ax.set_ylim(np.amin(self.obj.bias_linspace)-1, np.amax(self.obj.bias_linspace)+1)
    self.ax.set_xlim(np.amin(self.obj.slope_linspace)-1, np.amax(self.obj.slope_linspace)+1)
    self.ax.plot_surface(self.obj.slope_linspace, self.obj.bias_linspace,
                         self.obj.si_loss, cmap="Reds", linewidth=0, alpha=0.7)
    self.ax.set_title('Gradient Descent')
    sc = self.ax.scatter(self.obj.w[i], self.obj.b[i],
                         self.obj.loss[i]+0.1, color="b", s=100, alpha=1.0)
    self.ax.set_ylabel("Intercept")
    self.ax.set_xlabel("Slope")
    self.ax.set_zlabel("Loss")
    self.ax.view_init(45, -100)
    self.ax.legend((sc,), ("Gradient",), scatterpoints=1)

  def make_animation(self, func, file="/content/optimization.gif", interval=750, d3=False):
    if d3:
      fig = plt.figure()
      ax = plt.axes(projection='3d')
    else:
      fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    self.ax = ax
    ani = FuncAnimation(fig, func, frames=self.b.size-1, interval=interval)
    ani.save(file, writer="pillow")


if __name__ == "__main__":

    # ***Secant to tangent plot***
    # fig, ax = plt.subplots()
    # fig.set_figheight(6)
    # fig.set_figwidth(10)
    # animation = Animation(ax)
    # ani = FuncAnimation(fig, animation.secant_to_tangent, frames=121, interval=100)
    # ani.save("/content/secant_to_tangent.gif", writer="pillow")


    # ***slope and intercept animation***
    # GD = GradientDescent(0.4, 3, si_loss=True)
    # weight, bias = GD.optimize_slope_intercept(200)
    # GD.b.append(bias)
    # GD.w.append(weight)
    # GD.b = np.array(GD.b)
    # GD.w = np.array(GD.w)
    # GD.loss_space()
    # OptAni = OptimizationAnimation(GD.w, GD.b, np.amax(GD.y)+5, func=GD.plot_residuals, obj=GD)
    # OptAni.make_animation(OptAni.linear_regression, file="/content/slope_intercept_fit.gif", interval=80)
    # OptAni.make_animation(OptAni.gradient_descent, file="/content/slope_intercept_optimization.gif", interval=125, d3=True)


    # ***Intercept animation***
    # GD = GradientDescent(0.4, 3)
    # bias = GD.optimize_intercept()
    # GD.plot_residuals(0.4, bias, show=False, save="linear_regression.png")
    # GD.plot_residuals(0.4, GD.b[0], show=False, save="start_linear_regression.png")
    # GD.b.append(bias)
    # GD.b = np.array(GD.b)
    # GD.loss_space()
    # GD.intercept_losses = np.array(GD.intercept_losses)
    # GD.w = np.full((GD.b.size), 0.4)
    # OptAni = OptimizationAnimation(GD.w, GD.b, np.amax(GD.y)+5, func=GD.plot_residuals, obj=GD)
    # OptAni.make_animation(OptAni.linear_regression, file="/content/intercept_fit.gif", interval=150)
    # OptAni.lim = np.amax(GD.intercept_losses)+2
    # OptAni.make_animation(OptAni.intercept_derivative, file="/content/intercept_optimization.gif", interval=150)

    pass