<h1> PyTorch Agent Snake Solver </h1>

<h2> Overview </h2>

This project is a demonstration of the potential of artificial intelligence in playing and mastering video games. It leverages the power of PyTorch for the neural network and reinforcement learning components, combined with the Pygame library for game implementation and visualization. The result is an AI agent capable of playing the classic Snake game autonomously, learning and improving its performance through experience.

<h2> Motivation </h2>

The inspiration behind this project stems from my long-standing passion for video games and a keen interest in the burgeoning field of artificial intelligence. Video games have always fascinated me not only as a source of entertainment but also as complex systems that require strategic thinking, quick decision-making, and adaptability. With the rapid advancements in AI, I was curious to explore how these technologies can be applied to create intelligent agents capable of playing games.

Snake, being a simple yet challenging game, provides an ideal environment for experimenting with AI techniques. By developing this project, I aimed to delve deeper into reinforcement learning, understand its intricacies, and witness firsthand how an AI agent can evolve and improve its gameplay over time.

<h2> Features </h2>

1. AI Agent: Utilizes a deep Q-network (QNet) for decision making.
2. Training: The agent is trained using reinforcement learning principles, learning from its experiences in the game.
3. Game Environment: Built using Pygame, providing a visual representation of the game and the agent's actions.
4. Customizable Parameters: Various parameters such as learning rate, discount factor, and exploration rate can be tuned for optimal performance.
5. Real-time Visualization: Watch the AI play the game in real-time, showcasing its learning progress and strategy evolution.

<h2> Instructions to run the program on Windows </h2> 

1. Download Visual Studio Code. The official dowload page is available here: https://code.visualstudio.com/download
2. Make sure your Python version is 3.3 or higher. If not use command:
3. Move to folder Pytorch-Agent-Snake-Game-main
4. Create virtual environment. In terminal type:
```python
pip install virtualenv
```
then:
```python
python -m venv SnakeGame-env
```
If you use Linux type:
```python
sudo apt install python3-virtualenv
```
and:
```python
python -m venv SnakeGame-env
```
5. Then activate it by typing:
```python
SnakeGame-env\Scripts\activate
```
6. Install necessary packages using requirements.txt. Type in console:
```python
pip install -r requirements.txt
```
7. Run the program with command:
```python
python agent.py
```

<h2> Instructions to run the program on Unix/MacOS </h2> 

1. Download Visual Studio Code. The official dowload page is available here: https://code.visualstudio.com/download
2. Make sure your Python version is 3.3 or higher. If not use command:
3. Move to folder Pytorch-Agent-Snake-Game-main
4. Create virtual environment. In terminal type:
```python
python -m venv SnakeGame-env
```
5. Then activate it by typing:
```python
source /Scripts/activate
```
6. Install necessary packages using requirements.txt. Type in console:
```python
pip install -r requirements.txt
```
7. Run the program with command:
```python
python agent.py
```

<h2> Project Explanation </h2>
TBA

Project is implementation of behaviour approach by Patrick Loeber under MIT License
https://github.com/patrickloeber/snake-ai-pytorch/blob/main/LICENSE
