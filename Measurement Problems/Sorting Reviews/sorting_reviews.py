import pandas as pd
import math
import scipy.stats as stats


pd.set_option('display.width', 1881)
pd.set_option('display.max_columns', 1881)


# Sort by Wilson Lower Bound Score
def wilson_lower_bound(positive, n, confidence = 0.95):
    """
    Calculates wilson lower bound
    :param positive: Number of positives
    :param n: Number of samples
    :param confidence: Confidence level
    :return: Return wilson lower bound score
    """
    if n == 0:
        return 0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = positive / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def score_up_down_diff(up, down):
    return up - down


# Case Study
up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})

# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["up"] + x["down"]), axis=1)



comments.sort_values("wilson_lower_bound", ascending=False)