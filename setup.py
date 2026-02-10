from setuptools import setup, find_packages

setup(
    name="arduino-uno-q4gb-ai-robot",
    version="1.0.0",
    description="Arduino UNO Q4GB AI Robot with Object Detection, Camera Control, and Web Interface",
    author="Invention Convention 2026",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "onnxruntime>=1.15.0",
        "pyserial>=3.5",
        "flask>=2.3.0",
        "Pillow>=10.0.0",
        "tensorflow>=2.13.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "arduino-ai-robot=arduino_q4gb_ai_robot_complete_final.main_ai_robot:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)