from setuptools import setup, find_packages

setup(
    name="robot-arm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.1",
        "stable-baselines3>=2.0.0",
        "mujoco>=3.0.0",
        "tensorboard>=2.13.0",  # For training visualization
    ],
    python_requires=">=3.8",
)
