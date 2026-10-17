"""Random Forest strategy using the standard strategy interface."""

from .probability_strategy import ProbabilityStrategy


class RandomForestStrategy(ProbabilityStrategy):
    """Map Random Forest probabilities to target long-only exposure."""

    name = "random_forest"
    display_name = "Random Forest"
