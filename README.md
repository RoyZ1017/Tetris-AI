# Overview
The goal of this project was to create an AI that could play Tetris at an effective level. To do this I used both reinforcement learning and genetic algorithms to train the AI, comparing the two to see which method was most effective. 

# Requirements to Run
  - Python 3
  - Stable Baselines3
  - Neat-python
### Windows
  - ```pip install pygame```
  - ```pip install stable-baselines3[extra]```
  - ```pip install neat-python```

# How It Works
After experimenting with many different implementations of the Tetris AI a heuristic approach proved to be most successful. In this approach the AI takes in the game state, then returns 4 different weights representing the importance of 4 different heuristics. These 4 heuristics are: aggregate height, number of completed rows, number of holes, and the bumpiness (sum of the differences in height between all adjacent columns). We then analyze all possible board positions given our current block and determine a best move using the weights of the four heuristics given. The scoring equation used to determine the score of each board state is the following:

```score = aggregate height weight * aggregate height + completed rows weight * completed rows + holes weight  * number of holes + bumpiness weight * bumpiness```

After determining the best board state, the move which achieves this board state is then carried out. This is different from many other heuristic Tetris AI's as instead of having a static set of weights throughout the entire game, a dynamic weight is used instead which changes as the board progresses. This was done as I believed that this method would be more effective as the importance of actions change as you progress throughout the game. For example at the start of the game when the tower height is low, you may want to prioritize making the tower as flat as possible while minimizing holes to set up future tetrises. Whereas at the later stages of the game when the tower height is high you may want to prioritize minimizing tower height and clearing rows in order to survive. This idea sounded good in theory but may have caused some consistency issues which will be later discussed in the results section.

# Reinforcement Learning
### Representing Game State
Representing the Tetris game state proved rather difficult due to the complexity and number of combinations a tetris board could have. A standard 10x20 tetris game board can be represented as a matrix of 1's and 0's where 1 represents a block and 0 represents an empty block. The problem with this representation is the sheer number of possibilities the game board could have, having a total of 2^200 different possibilities (a little less as certain positions are unobtainable in a standard game of tetris). So in order for our model to learn effectively this game board had to be simplified. This was done by compacting the game board into 20 values ranging between 0 and 20. The first 10 represents the max height in each column and the next 10 represents the number of holes in each column. This way of representation may not provide all the information the original game board does, however it simplifies it into a much more manageable state while still providing most of the important information. To make our game state we can simply concatenate this simplified game board along with the current, held, and next blocks into a 1d vector in the following format:

```state = [height for each col, num holes in each col, curr block, held block, next 3 blocks]```

### Actions
The action which the AI would return would also vary and evolve as the representation of the game state changed. The action the AI would return started as a simple command e.g. shift block left, rotate block etc. However, it became evident pretty early on that this wasn't going to work. This was the case as there were too many combinations of actions that would lead to the same board, and each action was too far away from the reward, making it difficult to learn as a good action wouldn't be rewarded until much later on. This resulted in the development of a much more simplified action representation where the AI simply returned the rotation and column the block should be dropped in. This grouped action eliminates some fancy actions seen in modern Tetris such as slotting and t-spins, however allows for the AI to achieve immediate reward significantly improving the training stage. This was then changed to the current heuristic approach where the agent returns 4 different weights, however the idea of returning a rotation and a column continued to be used when trying to determine all possible board states resulting from the current block. This returned weight would then be carried out for the next four blocks as the AI only had information regarding the next four blocks, but also made training much easier as it reduced the chance that a good weight would produce minimal reward. 

### Reward
The reward function remains mainly untouched and is simply the standard tetris scoring function with a slight penalty for dying. The reward function is the represented by the following:
  - +10 for placing a block
  - +40 for clearing a single line
  - +100 for clearing 2 lines
  - +300 for clearing 3 lines
  - +1200 for clearing 4 lines
  - -5 for dying 

### Training
This model was trained for a total of 1 million total time steps using stable baslines3's PPO algorithm. More training would've been ideal, however I was unable to do so due to the changes made during the project and the time I had available. Throughout the training process steady growth was evident and I believe that this trend would continue as training progressed. 

# Genetic Algorithm
### Implementation
The overall structure of the genetic algorithm is quite similar to the reinforcement learning model as the same environment is used. Both have the same general inputs, outputs, and reward with only a few key differences.

### Fitness Function
The Fitness Function is the exact same as the reward function used in the reinforcement model, however the final fitness value is determined by the average over a 5 game window. 

### Training
This model was trained using the NEAT library with a population size of 50, an activation mutation rate of 0.1, and a survival threshold of 0.2. The model started off with 25 inputs, 4 outputs, and 0 hidden layers, and had a max generation of 50. It is worth mentioning that the model seemed to stagnate after generation 30 seeing little to no further growth. 

# Results
Overall both AI's were able to play Tetris at an effective level better than most humans. Both AI's can clear up to 1000 lines at most however it is somewhat inconsistent due to piece luck. The model trained using the NEAT algorithm was particularly unstable, sometimes dying in the early game. The reinforcement model also suffers from some inconsistency problems, but is much better in comparison to the NEAT model. This could be due to the fact that some board states weren't seen in the training phase resulting in poorly produced weights during the testing phase. However despite this inconsistency both models could still easily clear well over 100 lines venturing close to 500 and sometimes upwards of 900-1000. In the end both models were able to play Tetris at an effective level, but the winner between the two would have to be the reinforcement learning model due to its added consistency. 

# Next Steps
I believe that with further training and hyper parameter tuning, the AI can continue to improve and improve it's consistency issues. Allowing the AI to also calculate all possible board states for the next block(instead of just the current block) can also improve performance. This idea was originally tested, however it made the training process significantly slower, and due to the lack of time available, the idea was never carried out. 
