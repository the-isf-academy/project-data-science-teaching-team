# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

import numpy as np
import math
import random
from sklearn import datasets, linear_model

# +
def generate_counts_dict(data_list):
    data_dict = {}
    for element in data_list:
        if element in data_dict:
            data_dict[element] = data_dict[element] + 1
        else:
            data_dict[element] = 1
    return data_dict

# STUDENT WORK
def calculate_mean(list_of_numbers):
    """
    Returns the average (mean) of a list of numbers.
    input: list of ints (or floats)
    output: float
    """
    return sum (list_of_numbers)/len(list_of_numbers)

# STUDENT WORK
def is_even(number):
    """
    Returns True if the number is even. Otherwise, False.
    This is useful because the rule for calculating the median is different 
    when there are an even number of elements in the list.
    """
    return number % 2 == 0

def sort_list(unsorted_list):
    """
    Sorts the list of values and returns a list in ascending order (from lowest to highest)
    input: list of ints(or floats)
    output: list of ints(or floats)
    """
    return sorted(unsorted_list)

def calculate_median(sorted_list):
    """
    Takes the sorted list, returns the median of the list
    input: list of ints(or floats)
    output: integer (or float)
    """
    if is_even(len(sorted_list)):
        middle_pos = int(len(sorted_list) / 2)
        median = (sorted_list[middle_pos - 1] + sorted_list[middle_pos]) / 2.0
    else:
        middle_pos = int((len(sorted_list) / 2) + 0.5)
        median = sorted_list[middle_pos - 1]
    return median


# +
def draw(data_points_list, ax, jitter = False):
    """
    Plots the points on a matplotlib Axes ax. Data points list should be something like
    [[10, 30], [45, 55], [-20, 10]]. Optionally addes jitter to the list to make the magnitude
    of ordinal or categorical values more obvious in the plot.
    input: list, matplotlib Axes, optional boolean
    output: matplotlib Axes
    """
    x, y = zip(*data_points_list)
    if jitter:
        x = jitter_list(x, 0.1)
        y = jitter_list(y, 0.1)
    ax.scatter(x, y, alpha=0.5)
    return ax

def draw_line(slope, y_intercept, ax):
    """
    Draws a line with the given slope and y-intercept.
    If you have already drawn a scatter plot, will draw the line over that graph.
    input: int or float, int or float, matplotlib Axes
    output: matplotlib Axes
    """
    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [y_intercept + slope * xmin, y_intercept + slope * xmax])
    return ax


def add_loss(loss, ax):
    """
    Adds text to a matplotlib Axes ax to display the loss of a regression line. Text is
    position just outside the plot bounds in the top right corner.
    input: int or float, matplotlib Axes
    output: matplotlib Axes
    """
    ax.text(1,1,'loss: {}'.format(loss), bbox=dict(facecolor='red', alpha=0.9), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    return ax

def loss_one_point(point, m, b):
    point_x, point_y = point
    line_y = m*point_x + b
    residual = point_y - line_y
    return pow(residual,2)

def loss(m,b,data_points_list):
    sqerror = 0
    for point in data_points_list:
        sqerror += loss_one_point(point, m, b)
    return sqerror/len(data_points_list)

def create_graph(x_label, y_label):
    """
    Creates a pyplot graph with bottom left corner (0,0) and top right corner (1,1). Sets the axeses labels
    based on x_label and y_label. Returns the axes so it can be added to by future function calls.
    input: sting, string
    output: matplotlib Axes
    """
    fig = plt.figure()
    ax = fig.add_axes([.1,.1,.8,.8])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax

def animated_draw(slope, y_intercept, loss, ax, delay = .5):
    """
    Produced one step of a dynamic plot. If first step in the animation, plots a new line and adds a new text
    element to hold the loss text. In the following steps, updates the line and the loss text for the current
    iteration. Adds a delay after drawing in anticipation of being called again by a student's for loop.
    input: int or float, matplotlib Axes
    output: matplotlib Axes
    """
    if ax.lines:
        xmin, xmax = plt.xlim()
        for line in ax.lines:
            line.set_xdata([xmin, xmax])
            line.set_ydata([y_intercept + slope * xmin, y_intercept + slope * xmax])
        for text in ax.texts:
            text.set_text('loss: {}'.format(loss))
    else:
        xmin, xmax = plt.xlim()
        ax.plot([xmin, xmax], [y_intercept + slope * xmin, y_intercept + slope * xmax])
        ax.text(1,1,'loss: {}'.format(loss), bbox=dict(facecolor='red', alpha=0.9), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
    add_loss(loss, ax)
    ax.figure.canvas.draw()
    plt.pause(delay)
    return ax

def minloss(ax, data_points_list):
    lm = linear_model.LinearRegression()
#     lm.fit(df_two_col_clamp['grade'].values.reshape(-1,1),df_two_col_clamp['sm_time'].values)
    m = lm.coef_[0]
    b = lm.intercept_
    final_loss = loss(m, b, data_points_list)
    animated_draw(m, b, final_loss, ax)
    
def minlossmc(data_points_list):
    m_domain = (0,2)
    b_domain = (-3000, 3000)
    min_loss = loss(m_domain[0], b_domain[0], data_points_list)
    min_m = 0
    min_b = 0
    for i in range(1000):
        curr_m = random.uniform(m_domain[0], m_domain[1])
        curr_b = random.uniform(b_domain[0], b_domain[1])
        curr_loss = loss(curr_m, curr_b, data_points_list)
    #     animated_draw(curr_m, curr_b, curr_loss, ax)
        if curr_loss < min_loss:
            min_m = curr_m
            min_b = curr_b
            min_loss = curr_loss

    return min_m, min_b


# +
def clamp_point(x, domain):
    """
    Takes a value and a (min, max) domain and returns the value if it is inside the domain, the domain min if the value
    is smaller than the domain, or the domain max if the value is larger than the domain.
    input: int or float, (int or float, int or float)
    output: int or float
    """
    if x < domain[0]:
        return domain[0]
    if x > domain[1]:
        return domain[1]
    return x

def clamp_database(db, x_domain, y_domain):
    """
    Takes the values in df and clamps them to the specified x_domain. For example, let's say x_domain = (0, 100).
    If 4424 was in the df, it would be rewritten as 100. If -2 was in the df, it would be rewritten as 0.
    input: dataframe, tuple
    output: dataframe
    """
    for i in range(len(db)):
        x_clamp = clamp_point(db.iloc[i][0], x_domain)
        y_clamp = clamp_point(db.iloc[i][1], y_domain)
        db.iloc[i, 0] = x_clamp
        db.iloc[i, 1] = y_clamp
    return db
# -


