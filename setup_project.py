from setuptools import setup

setup(name="project_maze",
      version="0.1",
      url="https://github.com/tuzzer/gym-maze",
      author="Igor & Pedro",
      packages=["project"],

)

setup(name="gym_maze",
      version="0.4",
      url="https://github.com/tuzzer/gym-maze",
      author="Matthew T.K. Chan",
      license="MIT",
      packages=["project.gym_maze", "project.gym_maze.envs"],
      package_data = {
          "project.gym_maze.envs": ["maze_samples/*.npy"]
      },
      install_requires = ["gym", "pygame", "numpy"]
)