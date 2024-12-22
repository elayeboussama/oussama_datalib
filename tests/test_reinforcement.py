import pytest
import numpy as np
from src.datalib.reinforcement import basic_q_learning, sarsa

class MockEnvironment:
    """Mock environment for reinforcement learning tests."""
    def __init__(self):
        self.observation_space = MockSpace(10)
        self.action_space = MockSpace(4)

    def reset(self):
        return 0

    def step(self, action):
        return np.random.randint(0, 10), np.random.rand(), np.random.choice([True, False]), {}

class MockSpace:
    """Mock observation or action space."""
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)

@pytest.fixture
def mock_environment():
    """Fixture to provide a mock environment."""
    return MockEnvironment()

def test_basic_q_learning(mock_environment):
    """Test basic Q-learning."""
    q_table = basic_q_learning(mock_environment, episodes=10, alpha=0.1, gamma=0.9, epsilon=0.1)
    assert q_table.shape == (mock_environment.observation_space.n, mock_environment.action_space.n)

def test_sarsa(mock_environment):
    """Test SARSA algorithm."""
    q_table = sarsa(mock_environment, episodes=10, alpha=0.1, gamma=0.9, epsilon=0.1)
    assert q_table.shape == (mock_environment.observation_space.n, mock_environment.action_space.n)
