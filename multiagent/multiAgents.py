# multiAgents.py
# --------------
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


from util import manhattanDistance
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        fdist = []
        gdist = 0
        for x in range(0, newFood.width):
            for y in range(0, newFood.height):
                if newFood[x][y]:
                    fdist.append(util.manhattanDistance(newPos, (x, y)))
        newGhostStates = successorGameState.getGhostStates()
        ghostPosition = successorGameState.getGhostPositions()
        gdist = util.manhattanDistance(newPos, ghostPosition[0])
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "* YOUR CODE HERE *"

        # Declaring variables for later use.
        minimum_food_distance = 9999999
        food_count = 0
        old_food_count = 0
        ghost_list = []
        length_of_new_ghost_states = len(newGhostStates)
        matrix_dimension = ((newFood.width - 3) + (newFood.height - 3))

        # Finding the minimum distance using fdist list.
        for d in fdist:
            if d < minimum_food_distance and d != 0:
                minimum_food_distance = d

        if minimum_food_distance == 9999999:
            minimum_food_distance = 0

        # Finding the food count remaining to complete
        for x in range(0, newFood.width):
            for y in range(0, newFood.height):
                if newFood[x][y]:
                    food_count = food_count + 1
                check = currentGameState.hasFood(x, y)
                if check:
                    old_food_count = old_food_count + 1

        # Finding the distance of ghost from the new position everytime
        for g in newGhostStates:
            distance = manhattanDistance(newPos, g.getPosition())
            if distance == 0:
                return -2
            gdist = gdist + distance
            ghost_list.append(gdist)

        # Checking the ghost distance, if it's too far, set the value of ghost distance to zero
        food_count = abs(old_food_count - food_count)
        gdist = gdist / length_of_new_ghost_states
        new_ghost_dist = float(gdist) / matrix_dimension
        if gdist >= 5:
            new_ghost_dist = 0

        # Checking the ghost distance, if it's too far, set the value of ghost distance to zero
        for dist in ghost_list:
            if dist > 1:
                continue
            else:
                new_ghost_dist = -2

        new_food_dist = float(minimum_food_distance) / matrix_dimension
        game_state = currentGameState.getPacmanPosition()
        final_value = float((1 - new_food_dist)) + float(food_count) + float(new_ghost_dist)

        # Set the evaluation function based on food distance, ghost distance and food count.
        # Also don't allow pacman to be stationary.
        if newPos != game_state:
            return float(final_value)
        else:
            return float(final_value) - 0.4

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        score, move = self.minimax(0, gameState, 0, True)
        return move

    # Returns the score and action from the current game state
    def minimax(self, ag, gameState, depth, player):

        win_state, lose_state = gameState.isWin(), gameState.isLose()

        # Base case
        if win_state:
            return self.evaluationFunction(gameState), 0

        elif lose_state:
            return self.evaluationFunction(gameState), 0

        elif depth == self.depth:
            return self.evaluationFunction(gameState), 0

        # Set best and worst values according to agent
        if ag == 0:
            best_value = -1000000000
        else:
            best_value = 1000000000

        legal_moves, num_of_agents = gameState.getLegalActions(ag), gameState.getNumAgents() - 1
        bad_move, best_move = '', ''

        # If it's a minimizing player, check if the agent is less than the num of agents, if so
        # continue at the same depth or else move depth+1 and find the worst move
        if not player:
            for i in range(len(legal_moves)):
                if ag < num_of_agents:
                    score, move = self.minimax(ag + 1, gameState.generateSuccessor(ag, legal_moves[i]), depth, False)
                else:
                    score, move = self.minimax(0, gameState.generateSuccessor(ag, legal_moves[i]), depth + 1, True)
                if score < best_value:
                    best_value, bad_move = score, legal_moves[i]
            return best_value, bad_move
        # If it's a maximizing player, find the best move
        else:
            for i in range(len(legal_moves)):
                score, move = self.minimax(1, gameState.generateSuccessor(ag, legal_moves[i]), depth, False)
                if score > best_value:
                    best_value, best_move = score, legal_moves[i]
            return best_value, best_move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alpha_beta_search(gameState)

    # Terminal test checks if the entire depth of the game tree has been covered
    # or the game has been won/lost
    def terminal_test(self, state, depth):
        return depth == self.depth or state.isWin() or state.isLose()

    # Returns the next agent (can be pacman or ghost)
    def get_next_agent(self, state, agent):
        return (agent + 1) % state.getNumAgents()

    # Checks if the next agent is pacman
    def is_next_pacman(self, state, agent):
        return self.index == self.get_next_agent(state, agent)

    # Returns the best action for pacman by performing alpha beta pruning on the way
    def alpha_beta_search(self, state):
        return self.max_value(state, 0, 0, float('-inf'), float('inf'))

    # Returns the max value for pacman
    def max_value(self, state, agent, depth, alpha, beta):

        # If terminal state has been reached, returns utility value
        if self.terminal_test(state, depth):
            return self.evaluationFunction(state)

        next_action, value = '', float('-inf')
        # get the max value among the legal actions
        for action in state.getLegalActions(agent):
            value = max(self.min_value(state.generateSuccessor(agent, action),
                        self.get_next_agent(state, agent), depth, alpha, beta), value)
            # if value is greater than beta, prune
            if value > beta:
                return value
            # update alpha and next_action
            if value > alpha:
                alpha = value
                next_action = action

        # return action if the game tree has been traversed and we are back at root(pacman)
        # value otherwise
        return next_action if depth == 0 else value

    # Returns the min value for ghosts
    def min_value(self, state, agent, depth, alpha, beta):

        # If terminal state has been reached, returns utility value
        if self.terminal_test(state, depth):
            return self.evaluationFunction(state)

        # Use min_value function until ghosts are encountered
        value_function, value = self.min_value, float('inf')
        # Increase the depth if next agent is pacman. Also use max_value function now
        if self.is_next_pacman(state, agent):
            depth, value_function = depth + 1, self.max_value

        # get the max value among the legal actions
        for action in state.getLegalActions(agent):
            value = min(value_function(state.generateSuccessor(agent, action),
                        self.get_next_agent(state, agent), depth, alpha, beta), value)
            # if value is less than alpha, prune
            if value < alpha:
                return value
            # update beta
            beta = min(beta, value)

        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax_search(gameState)

    # Terminal test checks if the entire depth of the game tree has been covered
    # or the game has been won/lost
    def terminal_test(self, state, depth):
        return depth == self.depth or state.isWin() or state.isLose()

    # Returns the next agent (can be pacman or ghost)
    def get_next_agent(self, state, agent):
        return (agent + 1) % state.getNumAgents()

    # Checks if the next agent is pacman
    def is_next_pacman(self, state, agent):
        return self.index == self.get_next_agent(state, agent)

    # Returns the best action for pacman by performing expectimax search for stochastic moves
    def expectimax_search(self, state):
        return self.max_value(state, 0, 0)

    # Returns the max value for pacman
    def max_value(self, state, agent, depth):

        # If terminal state has been reached, returns utility value
        if self.terminal_test(state, depth):
            return self.evaluationFunction(state)

        # Gets the max value and action for all successors of this node
        value, action = max([(self.min_value(state.generateSuccessor(agent, action),
                                             self.get_next_agent(state, agent), depth), action)
                             for action in state.getLegalActions(agent)])

        # return action if the game tree has been traversed and we are back at root(pacman)
        # value otherwise
        return action if depth == 0 else value

    # Returns the expected min value for ghosts (based on averaging chances)
    def min_value(self, state, agent, depth):

        # If terminal state has been reached, returns utility value
        if self.terminal_test(state, depth):
            return self.evaluationFunction(state)

        # Use min_value function until ghosts are encountered
        value_function = self.min_value
        # Increase the depth if next agent is pacman. Also use max_value function now
        if self.is_next_pacman(state, agent):
            depth, value_function = depth + 1, self.max_value

        # Gets all values for all successors of this node
        values = [value_function(state.generateSuccessor(agent, action), self.get_next_agent(state, agent),
                                 depth) for action in state.getLegalActions(agent)]

        # returns the expected score
        return float(sum(values)) / len(values)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

