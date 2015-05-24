"""
File: predictivesimilarity.py
Author: Joseph Rollinson
Email: JTRollinson@gmail.com
Github: jrollinson
Description: ExpOps and stopping for the predictive similarity policy.
"""

from expops import expops


def stop(predict_correct, state, update_state, similarity_threshold,
         confidence_threshold):
    """Returns True if should stop, False otherwise.

    :predict_correct: Function that takes in an observation sequence and
    returns probability that the next observation will be 'correct'.
    :state: State of the world.
    :update_state: Function that returns updated state of the world.
    :similarity_threshold: Threshold on similarity.
    :confidence_threshold: Threshold on the confidence that too similar.
    :returns: True if should stop, False otherwise.

    """
    current_p_c = predict_correct(state)
    current_p_w = 1 - current_p_c

    p_too_close = 0.0

    if current_p_c > 0.0:
        state_after_correct = update_state(state, 1)
        p_c_after_c = predict_correct(state_after_correct)

        if abs(current_p_c - p_c_after_c) < similarity_threshold:
            p_too_close += current_p_c

    if current_p_w > 0.0:
        state_after_wrong = update_state(state, 0)
        p_c_after_w = predict_correct(state_after_wrong)
        if abs(current_p_c - p_c_after_w) < similarity_threshold:
            p_too_close += current_p_w

    return p_too_close > confidence_threshold


def expops_similarity(predict_correct, start_state, update_state,
                      similarity_threshold, confidence_threshold,
                      path_threshold):
    """Returns the expected number of learning opportunities required for the
    similarity threshold.

    :predict_correct: Function that takes in observation sequence and returns
    the probability that the next observation is correct.
    :similarity_threshold: Similarity Threshold.
    :confidence_threshold: Confidence Threshold.
    :path_threshold: Path Threshold.
    :returns: Expected number of learning opportunities.

    """

    def stop_f(self, state):
        """Stop function for policy

        :state: The student's state
        :returns: True if policy says to stop

        """
        return stop(predict_correct, state, update_state,
                    similarity_threshold, confidence_threshold)

    return expops(predict_correct, stop_f, start_state, update_state,
                  path_threshold)
