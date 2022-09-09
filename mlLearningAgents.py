# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

# Q-learning algorithm written by: Seyed Mohammad Reza Shahrestani, 16/04/2022

from __future__ import absolute_import
from __future__ import print_function

import random
from tkinter.messagebox import NO
from turtle import st
from xxlimited import foo

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0

        # Initialising the values
        self.setValues()


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    # Setting the initial values for the q-table
    # @param: self
    # @return: -
    def setValues(self):

        # Initialising the q-table
        self.q_table = []
        
        # q-table states
        self.states = []

        # Initialising the state values to None
        self.currentState = None
        self.previousState = None

        # Initialising the move values to None
        self.currentMove = None
        self.previousMove = None

        self.score = 0
        self.previousScore = None
        
        # Set of actions
        self.moves = ["North", "East", "South", "West"]

    # Reseting the values after each game in final()
    # @param: self
    # @return: -
    def resetValues(self):
        self.currentState = None
        self.previousState = None
        self.currentMove = None
        self.previousMove = None
        self.score = 0

    # Update the values after each action in getBestAction()
    # @param: self, LIST: the current state, STRING: the best action example: "North", "East", "South", "West"
    # @return: -
    def updateStates(self, currentState, bestAction):
        #update the states and moves
        self.previousState = self.currentState
        self.currentState = currentState
        self.previousMove = self.currentMove
        self.currentMove = bestAction

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # Remove Stop from legal actions
    # @param: self, state
    # @return: LIST: legal actions. example: ['North', 'South', 'West']
    def removeStopAction(self,state):
        legalActions = state.getLegalPacmanActions()
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
        return legalActions

    # Get the exact food position
    # Convert food table to index table
    # Loops through the grid and finds the location of the foods
    # @param: self, PACMAN GRID: food 
    # @return: LIST: position of foods. example: [(1, 1), (3, 3)]
    def getExactFoodPosition(self, food):
        foodPositionGrid = []
        for h in range(food.height):
            for w in range(food.width):
                if(food[w][h] == True):
                    foodPositionGrid.append((w,h))
        return foodPositionGrid

    # This function finds the best action based on their heighest in the states
    # @param: self, INT: the index of the state, LIST: legalActions
    # @return: STRING: the best action. example: "North"
    def findBestAction(self, stateIndex, legalActions):

        # initialising max with -inf value
        max = float('-inf')

        # Initialising best action with the last action in the legal actions
        bestAction = legalActions[-1]
        for i in range(len(self.q_table[stateIndex])):
            if(self.moves[i] in legalActions):
                if (self.q_table[stateIndex][i] > max):
                    max = self.q_table[stateIndex][i] 
                    bestAction = self.moves[i]

        return bestAction


    # Initialises the current state 
    # Finds the index of the state in the q table
    # Finds the legal moves of the pacman
    # returns the best next move for pacman based on the highest value
    # @param: self, state
    # @return: STRING: the best action. example: "South"
    def getBestAction(self, state):

        # generating the table
        food = state.getFood()
        
        pacmanPosition = [state.getPacmanPosition()]
        ghostPosition = state.getGhostPositions()
        foodPosition = self.getExactFoodPosition(food)
        
        data = [pacmanPosition, ghostPosition, foodPosition]
        initial_q_value = [0,0,0,0]
        
        self.states.append(data)
        self.q_table.append(initial_q_value)

        currentState = [pacmanPosition, ghostPosition, foodPosition]
        stateIndex = self.states.index(currentState)
        
        legalActions = self.removeStopAction(state)

        bestAction = self.findBestAction(stateIndex, legalActions)

        #update the states
        self.updateStates(currentState, bestAction)

        return bestAction

    # @param: self
    # @return: INT: the index of the current row, INT: the index of the current column. example: 2 
    def getCurrentPosition(self):
        currentRow = self.states.index(self.currentState)
        currentCol = self.moves.index(self.currentMove)
        return currentRow, currentCol

    # @param: self
    # @return: INT: the index of the previous row, INT: the index of the previous column. example: 1
    def getPreviousPosition(self):
        previousRow = self.states.index(self.previousState)
        previousCol = self.moves.index(self.previousMove)
        return previousRow, previousCol

    # This is the method used to train the model using q-learning algorithm and it updates the reward for the model
    # @param: self, state
    # @return: -
    def trainModel(self, state):

        legalActions = self.removeStopAction(state)
        alpha = self.getAlpha()
        gamma = self.getGamma()

        # Removes the unchanged moves
        if(self.previousMove != None):

            # Updates the reward
            reward = state.getScore() - self.score

            previousPositions = self.getPreviousPosition()
            previousRow = previousPositions[0]
            previousCol = previousPositions[1]

            currentPosition = self.getCurrentPosition()
            currentRow = currentPosition[0]
            currentCol = currentPosition[1]


            # Q_old(s, a) = Q_old(s, a) + learning_rate ( reward + gamma * MAX( Q_new(s', a') ) - Q_old(s, a) )
            self.q_table[previousRow][previousCol] = self.q_table[previousRow][previousCol] + alpha * (reward + gamma * self.q_table[currentRow][currentCol] - self.q_table[previousRow][previousCol])

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    # This function is called in every step
    # This function determines the final action that pacman should take
    # @param: self, state
    # @return: STRING: the best action example: "North", "East", "South", "West"
    def getAction(self, state: GameState) -> Directions:

        # Chooses which action to choose
        # Initially is set to random action
        action = self.getBestAction(state)

        # Update scores
        self.trainModel(state)
        self.score = state.getScore()
        self.previousScore = state.getScore()

        # Determens where the pacman should go
        return action

    # This function is called when a game finishes
    # Either pacman wins or looses
    # It resets the values and updates the q table with the new reward
    # @param: self, state
    # @return: -
    def final(self, state: GameState):

        reward = state.getScore() - self.score
        currentPosition = self.getCurrentPosition()
        currentRow = currentPosition[0]
        currentCol = currentPosition[1]
        self.resetValues()

        # Q(s, a) = Q(s, a) + ( learning_rate * reward )
        self.q_table[currentRow][currentCol] = self.q_table[currentRow][currentCol] + self.getAlpha() * reward


        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)

# Thanks for reading my code! :)