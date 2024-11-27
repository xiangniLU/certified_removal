import math

def calculate_eps_prime(gamma, C, lam, n):

    """
    gamma:lipschitz常数
    C：裁剪系数
    lam：正则化系数
    n:数据数量
    """
    if n <= 1:
        raise ValueError("n must be greater than 1 to avoid division by zero.")
    eps_ = (4 * gamma * C**2) / (lam**2 * (n - 1))
    return eps_


def calculate_c(delta):
    """
    delta:可以容忍隐私泄露的概率
    """
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be between 0 and 1 (0 < delta < 1) to make log(1.5/delta) meaningful.")
    c = math.sqrt(2 * math.log(1.5 / delta))
    return c


def calculate_eps(c, eps_prime, d):
    """
    c:调节参数
    d:参数规模
    """
    if d <= 0:
        raise ValueError("d must be greater than 0 to avoid division by zero.")
    eps = (c * eps_prime) / math.sqrt(d)
    return eps

def calculate_GAN_bound(std, eps, c):
    if c == 0:
        raise ValueError("c must be non-zero to avoid division by zero.")
    GAN_bound = (std * eps) / c
    return GAN_bound