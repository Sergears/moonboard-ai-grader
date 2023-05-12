# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:34:12 2021

@author: sa262627
"""


import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pickle
import my_neural_regressor
import my_neural_classifier
import os




st.title("Moonboard AI grader")
st.sidebar.header('User input parameters')

st.markdown('Moonboard is a standartised training wall for rock climbers. A boulder problem is created by choosing the allowed holds. It then gets graded and repeated by users worldwide.')
st.markdown('In 2017 setup, grades of diffuculcy range from 6a+ to 8b+. Use the sidebar to select a problem from the test set. The problem will be automatically graded by the already trained neural networks.')


@st.cache_data
def load_data(path, nrows=0):    
    if nrows == 0:
        data = pd.read_csv(path + '/2017_test_data.csv')
    else:
        data = pd.read_csv(path + '/2017_test_data.csv', nrows=nrows)
    return data


def user_input_features(data):
    """    

    Parameters
    ----------
    data : pandas dataframe
        problem data of shape (n_problems, 199) organized as an array of 
        hold_statuses (len 198) and user_grade (len 1)

    Returns
    -------
    hold_statuses : numpy array of (198, 1)
        hold_statuses for the selected problem.
    user_grade : int
        problem user grade.

    """
    
    # grade selector
    grade_names = ['all', '6a+', '6b', '6b+', '6c', '6c+', '7a', '7a+', 
                       '7b', '7b+', '7c', '7c+', '8a', '8a+', '8b', '8b+']
    grade_name = st.sidebar.select_slider('Difficulcy grade', grade_names)
    if grade_name == 'all':
        grade = -1
    else:
        grade = grade_names.index(grade_name) + 1
    if grade != -1:
        data = data[data['Grade'] == grade]
        
    # repeats selector
    max_repeats = max(data['Repeats'])
    min_repeats = st.sidebar.slider('Show only problems with repeats above', min_value=0, max_value=max_repeats, value=0)
    data = data[data['Repeats'] >= min_repeats]
    
    # select problem
    if len(data) > 1:
        problem_ind = st.sidebar.slider('Problem ind', min_value=0, max_value=len(data)-1, value=0)
    else:
        problem_ind = 0
    row = data.iloc[problem_ind]
    hold_statuses = np.array(row[0:198])
    hold_statuses = hold_statuses.reshape(198, 1)
    user_grade = int(row[-1])
    
    return hold_statuses, user_grade


def display_problem(path, hold_statuses, user_grade, forest_grade, NN_regr_grade, NN_class_probs):
    """
    Displays hold selection overlayed on the moonboard setup

    Parameters
    ----------
    hold_statuses : dataframe column of shape (198,1)
        array of holds in the problem: 1 if a hold is in the problem, 0 otherwise.
    user_grade : integer from 0 to 16
        user grade (ground truth) of the problem
    forest_grade : float
        grade prediction by the random forest model.
    NN_regr_grade : float
        grade prediction by the neural network regressor.
    NN_class_probs : float
        grade prediction by the neural network classifier.

    Returns
    -------
    None.

    """
    
    # convert hold_statuses to a 2D array (field) to be dispayed
    n_hold_rows = 18
    n_hold_cols = 11
    hold_letters = 'ABCDEFGHIJK'
    hold_heights = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 
                    '12', '13', '14', '15', '16', '17', '18']
    field = [[0 for _ in range(n_hold_cols)] for _ in range(n_hold_rows)]
    for i, hold_status in enumerate(hold_statuses):
        hold_row = i % n_hold_rows
        hold_col = (i - hold_row) // n_hold_rows
        if hold_status == 1:
            field[hold_row][hold_col] = 1

    # Create grid sub plots
    gs = gridspec.GridSpec(5, 2)
    gs.update(wspace=0.4, hspace=0.1)
    fig = plt.figure()  
    
    # moonboard setup picture
    ax_problem = plt.subplot(gs[:, 0])
    ax_problem.set_title('Selected problem')
    img = mpimg.imread(path + '/moon_setup_2017_edit.jpg')
    ax_problem.imshow(img, extent=[-0.5, 10.5, -0.5, 17.5])
    
    # overlay hold positions
    cmap = colors.ListedColormap(['white', 'blue'])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax_problem.imshow(field, cmap=cmap, norm=norm, origin='lower', alpha=0.5)
    ax_problem.set_xticks([i for i in range(n_hold_cols)])
    ax_problem.set_xticklabels(hold_letters)
    ax_problem.set_yticks([i for i in range(n_hold_rows)])
    ax_problem.set_yticklabels(hold_heights)
    spacing = 0.5
    minorLocator = MultipleLocator(spacing)
    minorLocator2 = MultipleLocator(spacing)
    ax_problem.yaxis.set_minor_locator(minorLocator)
    ax_problem.xaxis.set_minor_locator(minorLocator2)
    ax_problem.grid(which='minor')
    
    # vertical bar chart with ground truth and regressor predictions
    ax_regressors = plt.subplot(gs[0:2, 1])
    ax_regressors.set_title('AI grade prediction')    
    regressor_models = ['NN \n regressor', 'Random \n forest', 'Ground \n truth']
    grade_predictions = [NN_regr_grade, forest_grade, user_grade]
    x_tick_pos = [i for i in range(17)]
    y_tick_pos = [i for i, _ in enumerate(regressor_models)]
    grade_names_full = ['5+', '6a', '6a+', '6b', '6b+', '6c', '6c+', '7a', '7a+', 
                   '7b', '7b+', '7c', '7c+', '8a', '8a+', '8b', '8b+']    
    ax_regressors.barh(y_tick_pos, grade_predictions, color='r', height=0.2)    
    ax_regressors.set_yticks(y_tick_pos)
    ax_regressors.set_yticklabels(regressor_models)
    ax_regressors.set_xticks(x_tick_pos)
    ax_regressors.set_xticklabels([])
    ax_regressors.grid(axis='x')
    ax_regressors.set_xlim([0, 16.5])
    
    # horizontal bar chart with classifier prediction
    ax_classifier = plt.subplot(gs[2, 1])
    predicted_distr = NN_class_probs
    ax_classifier.bar(x_tick_pos, predicted_distr, color='r', width=0.8)
    ax_classifier.set_xlim([0, 16.5])
    ax_classifier.set_xticks(x_tick_pos)
    ax_classifier.set_xticklabels(grade_names_full)
    for label in ax_classifier.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax_classifier.set_yticks([])
    ax_classifier.set_ylabel('NN \n classifier', rotation=0, labelpad=30)
    ax_classifier.grid(axis='x')    
    
    fig.tight_layout()    
    st.pyplot(fig)


def print_grade_predictions(user_grade, forest_grade, NN_regr_grade, NN_class_probs):
    """
    print grade predictions in text (predictions are rounded to the nearest grade)

    Parameters
    ----------
    user_grade : integer from 0 to 16
        user grade (ground truth) of the problem
    forest_grade : float
        grade prediction by the random forest model.
    NN_regr_grade : float
        grade prediction by the neural network regressor.
    NN_class_probs : float
        grade prediction by the neural network classifier.

    Returns
    -------
    None.

    """
    grade_names_full = ['5+', '6a', '6A+', '6B', '6B+', '6C', '6C+', '7A', 
                       '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+', '8B', '8B+']  
    'User grade (ground truth):', grade_names_full[int(user_grade)]
    
    forest_grade_rounded = round(forest_grade)
    forest_grade_rounded = min(forest_grade_rounded, len(grade_names_full) - 1)
    forest_grade_rounded = max(0, forest_grade_rounded)
    'Random forest prediction:', grade_names_full[forest_grade_rounded]
        
    NN_regr_grade_rounded = round(NN_regr_grade)
    NN_regr_grade_rounded = min(NN_regr_grade_rounded, len(grade_names_full) - 1)
    NN_regr_grade_rounded = max(0, NN_regr_grade_rounded)
    'Neural network regressor prediction', grade_names_full[NN_regr_grade_rounded]
    
    NN_class_grade = np.argmax(NN_class_probs)
    'Neural network classifier prediction', grade_names_full[NN_class_grade]


def conver_normalized_grade_to_abs_grade(normalized_grade):
    """   
    Parameters
    ----------
    normalized_grade : float
        grade vaule compressed into the interval [0.25, 0.75] used for network learning

    Returns
    -------
    abs_grade: float
        absolute grade in the interval [0, 16].

    """
    abs_grade = (normalized_grade - 0.25) * 16 / 0.5
    return abs_grade


path = os.path.dirname(os.path.abspath(__file__))
data = load_data(path)
hold_statuses, user_grade = user_input_features(data)

# predict grade with random forest from sklearn
net_name = '/forest_model'
with open(path + net_name + '.pkl', 'rb') as input:
    forest_model = pickle.load(input)
normalized_grade = forest_model.predict(hold_statuses.T)[0]
forest_grade = conver_normalized_grade_to_abs_grade(normalized_grade)

# predict grade with my neural network regressor
net_name = '/nn_regressor_network'
file = open(path + net_name + '.pkl', 'rb')
parameters = pickle.load(file)
AL, caches = my_neural_regressor.L_model_forward(hold_statuses, parameters, 'relu', 'relu')
normalized_grade = AL[0,0]
NN_regr_grade = conver_normalized_grade_to_abs_grade(normalized_grade)

# predict grade with my neural network regressor
net_name = '/nn_classifier_network'
file = open(path + net_name + '.pkl', 'rb')
parameters = pickle.load(file)
AL, caches = my_neural_classifier.L_model_forward(hold_statuses, parameters, 'relu', 'sigmoid')
output = AL.T[0]
output_sum = sum(output)
NN_class_probs = np.array(output / output_sum)

display_problem(path, hold_statuses, user_grade, forest_grade, NN_regr_grade, NN_class_probs)
print_grade_predictions(user_grade, forest_grade, NN_regr_grade, NN_class_probs)