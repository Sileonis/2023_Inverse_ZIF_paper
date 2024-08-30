import numpy as np
import matplotlib.pyplot as plt

def plot_logD_trainSize_perMethod(frame1, frame2 = None, frame3 = None, method1_v_method2_stats = None, method1_v_method3_stats = None, label1 = '', label2 = '', label3 = '', on_off = 'False', x_min=0, x_max=75, y_min=0, y_max=10, 
               size='16', line=1.0, edge=2, axes_width = 2, tickWidth = 2, tickLength=12, 
            xLabel = '', yLabel ='', fileName = 'picture.png', marker_colors = ['y', 'g', 'r']):
    
    """ Plot the Mean (MAE) of logD (y-axis) to Size of Training Dataset (x-axis) for up to three methods 
        frame1 - 3:     A dataframe containing the follwing Columns:
                                                                    1.  sizeOfTrainingSet
                                                                    2.  averageError
                                                                    3.  stdErrorOfMeanError
        label1 - 3:     The name of the respective method used.
        on_off:         The value of frameon argument for pyplot.legend function.
        x_min:          The minimum value of x-axis
        x_max:          The maximum value of x-axis
        y_min:          The minimum value of y-axis
        y_max:          The maximum value of y-axis
        xLabel:         The label of x-axis
        yLabel:         The label of y-axis
        fileName:       The name under which the plot will be saved.
        marker_colors:  The colors that distinguish each method.
        """

    def round_no_zero(number, decimals = 1):        
        tmp_num = round(number,decimals)
        while tmp_num == 0:
            decimals += 1
            tmp_num = round(number,decimals)

        return round(number,decimals)

    # First Method
    x1 = frame1['sizeOfTrainingSet']
    y1 = frame1['averageError']
    error1 = frame1['stdErrorOfMeanError']

    division_point = int(len(x1)/2)

    x1_left  = x1.loc[0:division_point]
    x1_right = x1.loc[division_point + 1:]

    y1_left  = y1.loc[0:division_point]
    y1_right = y1.loc[division_point + 1:]

    error1_left  = error1.loc[0:division_point]
    error1_right = error1.loc[division_point + 1:]

    min_max_diff  = round_no_zero(max(y1_right) - min(y1_right), 1)
    low_y1_right  = round_no_zero(min(y1_right) - (min_max_diff / 2), 1)
    high_y1_right = round_no_zero(max(y1_right) + (min_max_diff / 2), 1)
    fourth_step   = round_no_zero((high_y1_right - low_y1_right) / 4, 1)

    plt.subplots(2,1)

    plt.subplot(2,1,1)
    plt.errorbar(x1_left, y1_left, yerr=error1_left, label=label1, ecolor='k', fmt='o', c=marker_colors[0], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
    plt.legend(loc='upper right', fontsize=15, frameon=on_off)

    plt.subplot(2,1,2)
    plt.errorbar(x1_right, y1_right, yerr=error1_right, label=label1, ecolor='k', fmt='o', c=marker_colors[0], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
    plt.legend(loc='upper right', fontsize=15, frameon=on_off)

    plt.xticks(np.arange(division_point, max(x1) + 5, 5.0))
    plt.yticks(np.arange(low_y1_right, high_y1_right, fourth_step))

    # Second Method
    if frame2 is not None:
        x2 = [x + 0.3 for x in frame2['sizeOfTrainingSet']]
        y2 = frame2['averageError']
        error2 = frame2['stdErrorOfMeanError']

        x2_left  = x2[0:division_point]
        x2_right = x2[division_point:]

        y2_left  = y2.loc[0:division_point]
        y2_right = y2.loc[division_point + 1:]

        error2_left  = error2.loc[0:division_point]
        error2_right = error2.loc[division_point + 1:]

        min_max_diff  = round_no_zero(max(y2_right) - min(y2_right), 1)
        low_y2_right  = round_no_zero(min(y2_right) - (min_max_diff / 2) ,1) 
        high_y2_right = round_no_zero(max(y2_right) + (min_max_diff / 2), 1)
        if high_y1_right > high_y2_right:
            high_y2_right = high_y1_right        
        fourth_step   = round_no_zero((high_y2_right - low_y2_right) / 4, 1)

        method1_v_method2_text = "\n".join((" ".join(("P-Value:     ", "{:.3e}".format(method1_v_method2_stats["pvalue"]))),
                                            " ".join(("Stat score:"  , "{:.3e}".format(method1_v_method2_stats["statistic"])))))

        plt.subplot(2,1,1)
        plt.errorbar(x2_left, y2_left, yerr=error2_left, label=label2, ecolor='k', fmt='o', c=marker_colors[1], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.errorbar([ ], [ ], None, label=method1_v_method2_text, linestyle='None')
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)

        plt.subplot(2,1,2)
        plt.errorbar(x2_right, y2_right, yerr=error2_right, label=label2, ecolor='k', fmt='o', c=marker_colors[1], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.errorbar([ ], [ ], None, label=method1_v_method2_text, linestyle='None')
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)

        plt.xticks(np.arange(division_point, max(x2) + 5, 5.0))
        plt.yticks(np.arange(low_y2_right, high_y2_right, fourth_step))

    # Third Method
    if frame3 is not None:
        x3 = [x + 0.4 for x in frame3['sizeOfTrainingSet']]
        y3 = frame3['averageError']
        error3 = frame3['stdErrorOfMeanError']

        method1_v_method3_text = "\n".join((" ".join(("P-Value:     ",    "{:.3e}".format(method1_v_method3_stats["pvalue"]))),
                                            " ".join(("Stat score:", "{:.3e}".format(method1_v_method3_stats["statistic"])))))

        x3_left  = x3[0:division_point]
        x3_right = x3[division_point:]

        y3_left  = y3.loc[0:division_point]
        y3_right = y3.loc[division_point + 1:]

        error3_left  = error3.loc[0:division_point]
        error3_right = error3.loc[division_point + 1:]

        min_max_diff  = round_no_zero(max(y3_right) - min(y3_right), 1)
        low_y3_right  = round_no_zero(min(y3_right) - (min_max_diff / 2), 1)
        high_y3_right = round_no_zero(max(y3_right) + (min_max_diff / 2), 1)
        if high_y1_right > high_y3_right:
            high_y3_right = high_y1_right
        if high_y2_right > high_y3_right:
            high_y3_right = high_y2_right
        fourth_step   = round_no_zero((high_y3_right - low_y3_right) / 4, 1)

        plt.subplot(2,1,1)
        plt.errorbar(x3_left, y3_left, yerr=error3_left, label=label3, ecolor='k', fmt='o', c=marker_colors[2], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.errorbar([ ], [ ], None, label=method1_v_method3_text, linestyle='None')
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)
    
        plt.subplot(2,1,2)
        plt.errorbar(x3_right, y3_right, yerr=error3_right, label=label2, ecolor='k', fmt='o', c=marker_colors[2], markersize=size, linewidth=line, markeredgecolor='k', markeredgewidth=edge)
        plt.errorbar([ ], [ ], None, label=method1_v_method2_text, linestyle='None')
        plt.legend(loc='upper right', fontsize=15, frameon=on_off)

        plt.xticks(np.arange(division_point, max(x3) + 5, 5.0))
        plt.yticks(np.arange(low_y3_right, high_y3_right, fourth_step))


    plt.tick_params(which='both', width=tickWidth)
    plt.tick_params(which='major', length=tickLength)

    plt.savefig(fileName, bbox_inches='tight')
    plt.show()