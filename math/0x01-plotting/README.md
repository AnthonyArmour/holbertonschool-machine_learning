Tasks
-----

### 0\. Line Graph

mandatory

Complete the following source code to plot `y` as a line graph:

*   `y` should be plotted as a solid red line
*   The x-axis should range from 0 to 10

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt
    
    y = np.arange(0, 11) ** 3
    
    # your code here
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/664b2543b48ef4918687.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210719T221057Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=66fd4f5830a815d6543291aff4d0a2688a040133a65bc76ff45be0438459f46a)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x01-plotting`
*   File: `0-line.py`

Done?! Help

×

#### Students who are done with "0. Line Graph"

### 1\. Scatter

mandatory

Complete the following source code to plot `x ↦ y` as a scatter plot:

*   The x-axis should be labeled `Height (in)`
*   The y-axis should be labeled `Weight (lbs)`
*   The title should be `Men's Height vs Weight`
*   The data should be plotted as magenta points

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt
    
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    
    # your code here
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/1b143961d254e65afa2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210719T221057Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=064272f5903a432e3313b70d22e99ae9f992c1ff18acbdc8dd7c7f9802bd1982)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x01-plotting`
*   File: `1-scatter.py`

Done?! Help

×

#### Students who are done with "1. Scatter"

### 2\. Change of scale

mandatory

Complete the following source code to plot `x ↦ y` as a line graph:

*   The x-axis should be labeled `Time (years)`
*   The y-axis should be labeled `Fraction Remaining`
*   The title should be `Exponential Decay of C-14`
*   The y-axis should be logarithmically scaled
*   The x-axis should range from 0 to 28650

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    
    # your code here
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/2b6334feb069ae1b6014.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210719T221057Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=f59e3918e58d554eeb1f5dd6ec3614889241212ef43e8144469878220add3337)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x01-plotting`
*   File: `2-change_scale.py`

Done?! Help

×

#### Students who are done with "2. Change of scale"

### 3\. Two is better than one

mandatory

Complete the following source code to plot `x ↦ y1` and `x ↦ y2` as line graphs:

*   The x-axis should be labeled `Time (years)`
*   The y-axis should be labeled `Fraction Remaining`
*   The title should be `Exponential Decay of Radioactive Elements`
*   The x-axis should range from 0 to 20,000
*   The y-axis should range from 0 to 1
*   `x ↦ y1` should be plotted with a dashed red line
*   `x ↦ y2` should be plotted with a solid green line
*   A legend labeling `x ↦ y1` as `C-14` and `x ↦ y2` as `Ra-226` should be placed in the upper right hand corner of the plot

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt
    
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    
    # your code here
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/39eac4e8c8eb71469784.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210719T221057Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=020cefb9efe7d8e07b95460e070febcd98caf1b31b421d60a09b5bd17a271b36)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x01-plotting`
*   File: `3-two.py`

Done?! Help

×

#### Students who are done with "3. Two is better than one"

### 4\. Frequency

mandatory

Complete the following source code to plot a histogram of student scores for a project:

*   The x-axis should be labeled `Grades`
*   The y-axis should be labeled `Number of Students`
*   The x-axis should have bins every 10 units
*   The title should be `Project A`
*   The bars should be outlined in black

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    
    # your code here
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/10a48ad296d16ee8fb63.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210719T221057Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=59591f0f6351d5c8733a4f089a768decdee65b2e8477c8af614aea4eef260c2b)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x01-plotting`
*   File: `4-frequency.py`

Done?! Help

×

#### Students who are done with "4. Frequency"

### 5\. All in One

mandatory

Complete the following source code to plot all 5 previous graphs in one figure:

*   All axis labels and plot titles should have a font size of `x-small` (to fit nicely in one figure)
*   The plots should make a 3 x 2 grid
*   The last plot should take up two column widths (see below)
*   The title of the figure should be `All in One`

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt
    
    y0 = np.arange(0, 11) ** 3
    
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180
    
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)
    
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)
    
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    
    # your code here
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/e58d423ffd060a779753.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210719T221057Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8a2dcc1c65409e0aa695d6e4fed15e16d0f17e483de85410f2ce14089d1e1594)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x01-plotting`
*   File: `5-all_in_one.py`

Done?! Help

×

#### Students who are done with "5. All in One"

### 6\. Stacking Bars

mandatory

Complete the following source code to plot a stacked bar graph:

*   `fruit` is a matrix representing the number of fruit various people possess
    *   The columns of `fruit` represent the number of fruit `Farrah`, `Fred`, and `Felicia` have, respectively
    *   The rows of `fruit` represent the number of `apples`, `bananas`, `oranges`, and `peaches`, respectively
*   The bars should represent the number of fruit each person possesses:
    *   The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
    *   Each fruit should be represented by a specific color:
        *   `apples` = red
        *   `bananas` = yellow
        *   `oranges` = orange (`#ff8000`)
        *   `peaches` = peach (`#ffe5b4`)
        *   A legend should be used to indicate which fruit is represented by each color
    *   The bars should be stacked in the same order as the rows of `fruit`, from bottom to top
    *   The bars should have a width of `0.5`
*   The y-axis should be labeled `Quantity of Fruit`
*   The y-axis should range from 0 to 80 with ticks every 10 units
*   The title should be `Number of Fruit per Person`

    #!/usr/bin/env python3
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    
    # your code here
    

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/10/8058e8f96e697612d50d.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210719%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210719T221057Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0ae6ffe3892efbd6d41d3690c7665149f554e296a5115425ae070b83402254c7)

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x01-plotting`
*   File: `6-bars.py`
