"""
File: masteryconf.py
Author: Joseph Rollinson
Email: JTRollinson@gmail.com
Github: jrollinson
Description: The expops algorithm using the mastery threshold policy.
"""

from expops import expops


def expops_mastery(predict_correct, predict_mastery, start_state, update_state,
                   mastery_threshold, path_threshold):
    """Calculates the expected number of learning opportunities required to
    mastery a subject.

    :predict_correct: Function that takes in state and returns
    the probability that the next observation is correct.
    :predict_mastery: Function that predicts mastery given state.
    :start_state: Start state for predictor.
    :updated_state: Function for updating state of the world.
    :mastery_threshold: Threshold to consider mastery.
    :path_threshold: Threshold before stopping path.
    :returns: Expected number of learning opportunities.

    """

    def stop_f(state):
        """Policy stop function

        :state: Current state of the student
        :returns: True if the policy should stop
        """
        return predict_mastery(state) >= mastery_threshold

    return expops(predict_correct, stop_f, start_state, update_state,
                  path_threshold)
