"""
File: expectedsimilarity.py
Author: Joseph Rollinson
Email: JTRollinson@gmail.com
Github: jrollinson
Description: ExpOps and stopping function for the expected similarity
policy.
"""

from expops import expops


def stop(predict_correct, state, update_state,
         similarity_threshold):
    """Expected similarity stopping function

    :predict_correct: Function that takes in an observation sequence and
    returns probability that the next observation will be 'correct'.
    :state: State of the world.
    :update_state: Function that returns updated state of the world.
    :similarity_threshold: Threshold on similarity.
    :returns: True if should stop, False otherwise

    """
    p_correct = predict_correct(state)

    expected_p_correct = 0.0

    if p_correct > 0.0:
        state_after_correct = update_state(state, 1)
        p_correct_after_c = predict_correct(state_after_correct)

        expected_p_correct += p_correct * p_correct_after_c

    if p_correct < 1.0:
        state_after_wrong = update_state(state, 0)
        p_correct_after_w = predict_correct(state_after_wrong)

        expected_p_correct += (1 - p_correct) * p_correct_after_w

    expected_similarity = abs(p_correct - expected_p_correct)
    return expected_similarity < similarity_threshold


def expops_expsim(predict_correct, start_state, update_state,
                  similarity_threshold, path_threshold):
    """
    Calculates the expected number of learning opportunities using expected
    similarity as the stopping policy.
    """
    def stop_f(state):
        """
        Returns true if should stop in given state.
        """
        return stop(predict_correct, state, update_state,
                    similarity_threshold)

    return expops(predict_correct, stop_f, start_state, update_state,
                  path_threshold)
