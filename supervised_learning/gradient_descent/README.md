[![Linkedin](https://i.stack.imgur.com/gVE0j.png) LinkedIn](https://www.linkedin.com/in/AnthonyArmoursProfile)

# Gradient Descent

This folder contains the code I wrote for a gradient descent explanation. The code includes gradient descent for linear regression and the animations that represent the optimization process.

## Dependencies
| Library/Framework  | Version |
| ------------------ | ------- |
| Python             | ^3.7.3  |
| numpy              | ^1.19.5 |
| matplotlib         | ^3.4.3  |

## [Gradient Descent Animation Scripts](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/gradient_descent.py "Gradient Descent Animation Scripts")

``` python
if __name__ == "__main__":

    # Secant to tangent plot
    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    animation = Animation(ax)
    ani = FuncAnimation(fig, animation.secant_to_tangent, frames=121, interval=100)
    ani.save("./animations/secant_to_tangent.gif", writer="pillow")
```

---
![gif](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/animations/secant_to_tangent.gif)
---

``` python
if __name__ == "__main__":

    # slope and intercept animation
    GD = GradientDescent(0.4, 3, si_loss=True)
    weight, bias = GD.optimize_slope_intercept(200)
    GD.b.append(bias)
    GD.w.append(weight)
    GD.b = np.array(GD.b)
    GD.w = np.array(GD.w)
    GD.loss_space()
    OptAni = OptimizationAnimation(GD.w, GD.b, np.amax(GD.y)+5, func=GD.plot_residuals, obj=GD)
    OptAni.make_animation(OptAni.linear_regression, file="./animations/slope_intercept_fit.gif", interval=80)
    OptAni.make_animation(OptAni.gradient_descent, file="./animations/slope_intercept_optimization.gif", interval=125, d3=True)
```

---
![gif](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/animations/slope_intercept_fit.gif)
---

---
![gif](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/animations/slope_intercept_optimization.gif)
---

``` python
if __name__ == "__main__":

    # Intercept animation
    GD = GradientDescent(0.4, 3)
    bias = GD.optimize_intercept()
    GD.plot_residuals(0.4, bias, show=False, save="linear_regression.png")
    GD.plot_residuals(0.4, GD.b[0], show=False, save="start_linear_regression.png")
    GD.b.append(bias)
    GD.b = np.array(GD.b)
    GD.loss_space()
    GD.intercept_losses = np.array(GD.intercept_losses)
    GD.w = np.full((GD.b.size), 0.4)
    OptAni = OptimizationAnimation(GD.w, GD.b, np.amax(GD.y)+5, func=GD.plot_residuals, obj=GD)
    OptAni.make_animation(OptAni.linear_regression, file="./animations/intercept_fit.gif", interval=150)
    OptAni.lim = np.amax(GD.intercept_losses)+2
    OptAni.make_animation(OptAni.intercept_derivative, file="./animations/intercept_optimization.gif", interval=150)
```

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/animations/linear_regression.png)
---

---
![image](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/animations/start_linear_regression.png)
---


---
![gif](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/animations/intercept_fit.gif)
---

---
![gif](https://github.com/AnthonyArmour/holbertonschool-machine_learning/blob/master/supervised_learning/gradient_descent/animations/intercept_optimization.gif)
---