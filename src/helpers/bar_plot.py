import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size": 15})

def plot_robot_barplots(data_matrix, title=''):
    barWidth = 0.25
    fig, ax = plt.subplots(figsize=(12, 4))
    # Set position of bar on X axis
    br1 = np.arange(len(data_matrix[0]))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, data_matrix[0], color ='salmon', width=barWidth,
            edgecolor='grey', label='30 Babbling points')
    plt.bar(br2, data_matrix[1], color ='mediumturquoise', width=barWidth,
            edgecolor='grey', label='100 Babbling points')
    plt.bar(br3, data_matrix[2], color ='cornflowerblue', width=barWidth,
            edgecolor='grey', label='150 Babbling points')

    # Adding annotations on top of the bars
    for i, values in enumerate(data_matrix):
        for j, value in enumerate(values):
            plt.text(br1[j] + i * barWidth, value + 0.2, f'{value}', ha='center', va='bottom', color='black')

    # Adding Xticks
    plt.xlabel('Iterations', fontweight='bold', fontsize=15)
    plt.ylabel('Amount of found nodes', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(data_matrix[0]))],
               ['Initial', '1st. iter.', '2nd. iter.', '3rd. iter.'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.savefig(f'{title}.pdf', format='pdf')
    plt.show()

def plot_robot_heatmaps(robot_data, time_vector, feature_labels, cmap='viridis', title=''):
    """
    Plot heatmaps for each robot's feature.

    Parameters:
    - robot_data: Dictionary containing data for each robot
    - time_vector: List or array containing time values
    - feature_labels: List or array containing feature labels
    - cmap: Colormap to use for the heatmaps (default is 'viridis')
    - title: Title for the overall plot
    """
    # Find the overall minimum and maximum values across all data
    overall_min = min(np.min(data[0]) for data in robot_data.values())
    overall_max = max(np.max(data[0]) for data in robot_data.values())

    # Create subplots with 1 row and 'n' columns, where 'n' is the number of robots
    num_robots = len(robot_data)
    fig, axes = plt.subplots(1, num_robots, figsize=(3.5 * num_robots, 3))

    for i, (robot, data) in enumerate(robot_data.items()):
        # Extract data for the current robot
        data_array = np.array(data[0])

        # Plot heatmap for the current robot with consistent scale
        heatmap = axes[i].imshow(data_array, cmap=cmap, vmin=overall_min, vmax=overall_max)

        # Set axis labels based on input feature_labels and time_vector
        axes[i].set_xticks(np.arange(len(time_vector)))
        axes[i].set_xticklabels(time_vector)
        axes[i].set_yticks(np.arange(len(feature_labels)))
        axes[i].set_yticklabels(feature_labels)

        # Annotate each square with the respective value and increase font size
        for row in range(len(feature_labels)):
            for col in range(len(time_vector)):
                axes[i].text(col, row, f'{data_array[row, col]:.2f}', ha='center', va='center', color='w', fontsize=20)

        # Add x-axis label for each subplot
        if i == 1:
            axes[i].set_xlabel('Amount of babbling points', fontweight='bold', fontsize=20, labelpad=30)
        axes[i].set_title(robot)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Add a single colorbar to the right of heatmaps
    cbar = fig.colorbar(heatmap, ax=axes, orientation='vertical', pad=0.05, fraction=0.05)

    # Add a title to the overall plot
    fig.suptitle(title, fontsize=20, fontweight='bold')
    plt.savefig(f'{title}.pdf', format='pdf')
    # Show the plot
    plt.show()

if __name__ == "__main__":

    # Example data structure with signals matrices
    data_babbling_time = {
        'QTrobot': [
            [[1.686, 1.348, 1.112],
             [4.075, 3.877, 4.110],
             [6.017, 5.752, 6.977]]
        ],
        'NAO': [
            [[0.818, 0.825, 0.811],
             [2.830, 2.890, 2.813],
             [4.589, 4.593, 4.639]]
        ],
        'Freddy': [
            [[2.729, 2.325, 0],
             [8.832, 8.423, 0],
             [12.074, 12.468, 0]]
        ]
    }

    babbled_points = [30, 100, 150]
    feature_labels = ['Left', 'Right', 'Head']

    # plot_robot_heatmaps(data_babbling_time, babbled_points, feature_labels, "plasma", "Babbling Time in minutes for each robot")

    data_iterations_time = {
        'QTrobot': [
            [[8.742, 54.489, 116.001],
             [33.439, 32.558, 28.568],
             [11.833, 14.516, 12.851],
             [10.862, 12.224, 10.590]]
        ],
        'NAO': [
            [[10.113, 89.601, 171.253],
             [70.694, 78.652, 56.398],
             [32.635, 27.574, 28.418],
             [15.181, 16.762, 18.417]]
        ],
        'Freddy': [
            [[35.306, 634.695, 517.891],
             [86.568, 94.745, 57.675],
             [22.714, 30.151, 23.506],
             [9.244, 16.542, 8.351]]
        ]
    }

    babbled_points = [30, 100, 150]
    feature_labels = ['Initial', '1st it.', '2nd it.', '3rd it.']
    plot_robot_heatmaps(data_iterations_time, babbled_points, feature_labels, "plasma", "Time to iterate over the Graphs in minutes")

    QTrobot_data = np.array([[2286, 5315, 6869],
                             [4631, 6682, 8039],
                             [5032, 7047, 8267],
                             [5260, 7387, 8477]])
    NAO_data = np.array([[3874, 9275, 11622],
                         [6722, 11335, 13283],
                         [7803, 11989, 13955],
                         [8404, 12307, 14197]])
    Freddy_data = np.array([[20526, 56512, 44675],
                            [412 + 1021 + 2050 + 2566 + 3059 + 3422 + 396 + 994 + 1934 + 2464 + 3128 + 3574,
                             397 + 1393 + 4939 + 6563 + 8065 + 8943 + 382 + 1317 + 4715 + 6209 + 7625 + 8605,
                             257 + 930 + 3486 + 4763 + 5842 + 6614 + 258 + 908 + 3298 + 4451 + 5421 + 6162],
                            [415 + 1030 + 2089 + 2653 + 3251 + 3691 + 396 + 1012 + 1980 + 2546 + 3279 + 3782,
                             397 + 1393 + 4953 + 6604 + 8192 + 9150 + 382 + 1318 + 4749 + 6289 + 7737 +  8769,
                             257 + 930 + 3497 + 4796 + 5914 + 6717 + 258 + 912 + 3305 + 4485 + 5513 + 6275],
                            [417 + 1035 + 2110 + 2710 + 3319 + 3775 + 399 + 1027 + 2002 + 2606 + 3371 + 3942,
                             397 + 1393 + 4987 + 6658 + 8294 + 9306 + 382 + 1318 + 4761 + 6317 + 7800 + 8885,
                             257 + 930 + 3499 + 4803 + 5921 + 6729 + 258 + 912 + 3314 + 4525 + 5562 + 6355]])


    plot_robot_barplots(QTrobot_data.T, "Amount of nodes collected in all Graphs for QTrobot per iteration")
    plot_robot_barplots(NAO_data.T, "Amount of nodes collected in all Graphs for NAO per iteration")
    plot_robot_barplots(Freddy_data.T, "Nodes collected for all Graphs Freddy per iteration")
