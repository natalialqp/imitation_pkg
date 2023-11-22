import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class RobotSimulation:
    def __init__(self, list1, list2, list3):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.base = [[(0., 0., 0)]] * len(list1)
        self.num_frames = len(list1)
        self.max_range = None

        # Create a 3D plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_chain(self, chain, color='b'):
        x, y, z = zip(*chain)
        self.ax.plot(x, y, z, color=color, marker='o', linestyle='-', markersize=5)

    def update(self, frame):
        self.ax.cla()  # Clear the previous plot
        self.plot_chain(self.list1[frame], color='b')
        self.plot_chain(self.list2[frame], color='g')
        self.plot_chain(self.list3[frame], color='r')
        self.plot_chain(self.base[frame], color='orange')
        self.ax.set_xlabel('X [cm]')
        self.ax.set_ylabel('Y [cm]')
        self.ax.set_zlabel('Z [cm]')
        self.ax.set_title(f'Frame {frame + 1}')

        # Find the maximum range of coordinates across all lists for the current frame
        all_lists = [self.list1[frame], self.list2[frame], self.list3[frame], self.base[frame]]
        self.max_range = max(
            max(max(p[i] for p in lst) for i in range(3)) for lst in all_lists)

        # Set equal scales for all axes by adjusting the limits
        self.ax.set_xlim(-self.max_range, self.max_range)
        self.ax.set_ylim(-self.max_range, self.max_range)
        self.ax.set_zlim(0, self.max_range)

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.num_frames, interval=1000)
        # If you want to save the animation as a video file (e.g., mp4)
        # You need to have ffmpeg or another video writer installed.
        # ani.save('animation.mp4', writer='ffmpeg', fps=1)
        # Show the animation (Note: The animation will not display correctly in some IDEs, use plt.show() instead)
        plt.show()
