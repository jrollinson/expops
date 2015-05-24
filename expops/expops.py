"""
File: expops.py
Author: Joseph Rollinson
Email: jtrollinson@gmail.com
Description: Expected Learning Opportunity functions.
"""


def expops(predict_correct, stop_f, start_state, update_state, path_threshold,
           max_len=100):
    """A higher order function that returns the expected number of learning
    opportunities required to before stop_f(path) is true.

    :predict_correct: Function that takes in observation sequence and returns
    the probability that the next observation is correct.
    :stop_f: Function with argument path that returns whether to stop
    or not.
    :start_state: Starting state of the world for the predictor.
    :updated_state: Function for updating the state of the world.
    :path_threshold: Probability threshold before stopping down a path.
    :returns: Approximate expected number of learning opportunities required.

    """
    def inner_expops(state, p_path, length):
        """Recursive inner function of expops."""

        # We do p_path threshold first so that we don't attempt to deal with
        # state when p_path = 0
        if (p_path < path_threshold) or (length >= max_len) or stop_f(state):
            return 0.0
        else:

            p_correct = predict_correct(state)
            assert 0.0 <= p_correct <= 1.0, "P(correct) + {}".format(p_correct)

            if p_correct > 0.0:
                p_path_and_c = p_path * p_correct
                state_after_correct = update_state(state, 1)
                expops_given_c = inner_expops(state_after_correct,
                                              p_path_and_c,
                                              length + 1)
            else:
                expops_given_c = 0.0

            if p_correct < 1.0:
                p_path_and_w = p_path * (1 - p_correct)
                state_after_wrong = update_state(state, 0)
                expops_given_w = inner_expops(state_after_wrong, p_path_and_w,
                                              length + 1)
            else:
                expops_given_w = 0.0

            return (1 +
                    (p_correct * expops_given_c) +
                    ((1 - p_correct) * expops_given_w))

    return inner_expops(start_state, 1.0, 0)
