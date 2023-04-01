import math
from scipy.stats import norm
from scipy import optimize
import logging
from pricing.options.vanilla import OptionPricing

# Set logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d: %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


# Pricing for a Barrier option on a asset that provides a yield at rate q
class BarrierOption:
    def __init__(self, option_type, barrier_type, s0, k, t, r, q, v, h):
        """
        :param option_type: 'c' or 'p'
        :param barrier_type: 'down-in', 'down-out', 'up-in' or 'up-out'
        :param s0: current asset price
        :param k: option strike price
        :param t: time to expiration of option
        :param r: continuously compounded risk-free rate of interest per annum
        :param q: continuously compounded yield rate provided by the asset
        :param v: implied volatility of the option
        :param h: barrier level
        """
        self.option_type = option_type
        self.barrier_type = barrier_type
        self.s0 = s0
        self.k = k
        self.r = r
        self.q = q
        self.t = t
        self.v = v
        self.h = h
        self._lambda = (r - q + v**2 / 2) / v**2
        self._y = math.log(h**2 / (s0 * k)) / (v * math.sqrt(t)) + self._lambda * v * math.sqrt(t)
        self._x1 = math.log(s0 / h) / (v * math.sqrt(t)) + self._lambda * v * math.sqrt(t)
        self._y1 = math.log(h / s0) / (v * math.sqrt(t)) + self._lambda * v * math.sqrt(t)
        self._price = self._get_price(option_type, barrier_type, s0, k, t, r, q, v, h)

    def __repr__(self):
        return f"BarrierOption(option_type='{self.option_type}', barrier_type='{self.barrier_type}', s0={self.s0}, " \
               f"k={self.k}, t={self.t}, r={self.r}, q={self.q}, v={self.v}, h={self.h})"

    @property
    def price(self):
        return self._price

    def _get_price(self, option_type, barrier_type, s0, k, t, r, q, v, h):
        if option_type == 'c':
            c_di, c_do = self._get_call_di_do_price(s0, k, t, r, q, v, h)
            c_ui, c_uo = self._get_call_ui_uo_price(s0, k, t, r, q, v, h)
            if barrier_type == 'down-in':
                return c_di
            elif barrier_type == 'down-out':
                return c_do
            elif barrier_type == 'up-in':
                return c_ui
            elif barrier_type == 'up-out':
                return c_uo
        if option_type == 'p':
            p_di, p_do = self._get_put_di_do_price(s0, k, t, r, q, v, h)
            p_ui, p_uo = self._get_put_ui_uo_price(s0, k, t, r, q, v, h)
            if barrier_type == 'down-in':
                return p_di
            elif barrier_type == 'down-out':
                return p_do
            elif barrier_type == 'up-in':
                return p_ui
            elif barrier_type == 'up-out':
                return p_uo
        return None

    def _get_call_di_do_price(self, s0, k, t, r, q, v, h):
        c = OptionPricing('c', s0, k, t, r, q, v).price
        _lambda, y, x1, y1 = self._lambda, self._y, self._x1, self._y1
        if h <= k:
            c_di = s0 * math.exp(-q * t) * (h / s0) ** (2 * _lambda) * norm.cdf(y) - \
                   k * math.exp(-r * t) * (h / s0) ** (2 * _lambda - 2) * norm.cdf(y - v * math.sqrt(t))
            c_do = c - c_di
            return c_di, c_do
        else:
            c_do = s0 * norm.cdf(x1) * math.exp(-q * t) - k * math.exp(-r * t) * norm.cdf(x1 - v * math.sqrt(t)) - \
                   s0 * math.exp(-q * t) * (h / s0) ** (2 * _lambda) * norm.cdf(y1) + \
                   k * math.exp(-r * t) * (h / s0) ** (2 * _lambda - 2) * norm.cdf(y1 - v * math.sqrt(t))
            c_di = c - c_do
            return c_di, c_do

    def _get_put_di_do_price(self, s0, k, t, r, q, v, h):
        p = OptionPricing('p', s0, k, t, r, q, v).price
        _lambda, y, x1, y1 = self._lambda, self._y, self._x1, self._y1
        if h <= k:
            p_di = -s0 * norm.cdf(-x1) * math.exp(-q * t) + k * math.exp(-r * t) * norm.cdf(-x1 + v * math.sqrt(t)) + \
                   s0 * math.exp(-q * t) * (h / s0) ** (2 * _lambda) * (norm.cdf(y) - norm.cdf(y1)) - \
                   k * math.exp(-r * t) * (h / s0) ** (2 * _lambda - 2) * (norm.cdf(y - v * math.sqrt(t)) -
                                                                           norm.cdf(y1 - v * math.sqrt(t)))
            p_do = p - p_di
            return p_di, p_do
        else:
            p_di = p
            p_do = 0
            return p_di, p_do

    def _get_call_ui_uo_price(self, s0, k, t, r, q, v, h):
        c = OptionPricing('c', s0, k, t, r, q, v).price
        _lambda, y, x1, y1 = self._lambda, self._y, self._x1, self._y1
        if h <= k:
            c_ui = c
            c_uo = 0
            return c_ui, c_uo
        else:
            c_ui = s0 * norm.cdf(x1) * math.exp(-q * t) - k * math.exp(-r * t) * norm.cdf(x1 - v * math.sqrt(t)) - \
                   s0 * math.exp(-q * t) * (h / s0) ** (2 * _lambda) * (norm.cdf(-y) - norm.cdf(-y1)) + \
                   k * math.exp(-r * t) * (h / s0) ** (2 * _lambda - 2) * (norm.cdf(-y + v * math.sqrt(t)) -
                                                                           norm.cdf(-y1 + v * math.sqrt(t)))
            c_uo = c - c_ui
            return c_ui, c_uo

    def _get_put_ui_uo_price(self, s0, k, t, r, q, v, h):
        p = OptionPricing('p', s0, k, t, r, q, v).price
        _lambda, y, x1, y1 = self._lambda, self._y, self._x1, self._y1
        if h <= k:
            p_uo = -s0 * norm.cdf(-x1) * math.exp(-q * t) + k * math.exp(-r * t) * norm.cdf(-x1 + v * math.sqrt(t)) + \
                   s0 * math.exp(-q * t) * (h / s0) ** (2 * _lambda) * norm.cdf(-y1) - \
                   k * math.exp(-r * t) * (h / s0) ** (2 * _lambda - 2) * norm.cdf(-y1 + v * math.sqrt(t))

            p_ui = p - p_uo
            return p_ui, p_uo
        else:
            p_ui = -s0 * math.exp(-q * t) * (h / s0) ** (2 * _lambda) * norm.cdf(-y) + \
                   k * math.exp(-r * t) * (h / s0) ** (2 * _lambda - 2) * norm.cdf(-y + v * math.sqrt(t))
            p_uo = p - p_ui
            return p_ui, p_uo


# Pricing for a Asian option on a asset that provides a yield at rate q
class AsianOption:
    def __init__(self, option_type, s0, k, t, r, q, v):
        self.option_type = option_type
        self.s0 = s0
        self.k = k
        self.r = r
        self.q = q
        self.t = t
        self.v = v
        self._m1 = self._get_m1(s0, t, r, q)
        self._m2 = self._get_m2(s0, t, r, q, v)
        self._f = self._m1
        self._v = math.sqrt(1 / t * math.log(self._m2 / self._m1 ** 2))
        self._d1 = (math.log(self._f / k) + self._v ** 2 / 2 * t) / (self._v * math.sqrt(t))
        self._d2 = self._d1 - self._v * math.sqrt(t)
        self._price = self._get_price(option_type, self._f, k, t, r, self._d1, self._d2)

    def __repr__(self):
        return f"AsianOption(option_type='{self.option_type}', s0={self.s0}, k={self.k}, t={self.t}, r={self.r}, " \
               f"q={self.q}, v={self.v})"

    @property
    def price(self):
        return self._price

    @staticmethod
    def _get_price(option_type, f0, k, t, r, d1, d2):
        if option_type == 'c':
            return f0 * math.exp(-r * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
        if option_type == 'p':
            return k * math.exp(-r * t) * norm.cdf(-d2) - f0 * math.exp(-r * t) * norm.cdf(-d1)
        return None

    @staticmethod
    def _get_m1(s0, t, r, q):
        if r == q:
            return s0
        return (math.exp((r - q) * t) - 1) / ((r - q) * t) * s0

    @staticmethod
    def _get_m2(s0, t, r, q, v):
        if r == q:
            m2 = 2 * s0 ** 2 * (math.exp(v**2 * t) - 1 - t * v ** 2) / (t ** 2 * v ** 4)
        else:
            m2 = 2 * math.exp((2 * (r - q) + v**2) * t) * s0 ** 2 / ((r - q + v**2) * (2 * r - 2 * q + v**2) * t ** 2) + \
                2 * s0 ** 2 / ((r - q) * t ** 2) * (1 / (2 * (r - q) + v ** 2) - math.exp((r - q) * t) / (r - q + v ** 2))
        return m2


# Pricing for a Spread option using Kirk's Approximation
class SpreadOption:
    def __init__(self, option_type, s1, s2, k, t, r, q, v1, v2, corr):
        """
        :param option_type: 'c' or 'p'
        :param s1: current price of the first asset
        :param s2: current price of the second asset
        :param k: option strike price
        :param t: time to expiration of option
        :param r: continuously compounded risk-free rate of interest per annum
        :param q: continuously compounded yield rate provided by the asset
        :param v1: implied volatility of the first asset
        :param v2: implied volatility of the second asset
        :param corr: correlation of the two assets
        """
        self.option_type = option_type
        self.s1 = s1
        self.s2 = s2
        self.k = k
        self.r = r
        self.q = q
        self.t = t
        self.v1 = v1
        self.v2 = v2
        self.corr = corr
        self.v = self._get_v(s2, k, v1, v2, corr)
        self._price = (s2 + k) * OptionPricing(option_type, s1 / (s2 + k), 1, t, r, q, self.v).price

    def __repr__(self):
        return f"SpreadOption(option_type='{self.option_type}', s1={self.s1}, s2={self.s2}, k={self.k}, t={self.t}, " \
               f"r={self.r}, q={self.q}, v1={self.v1}, v2={self.v2}, corr={self.corr})"

    @property
    def price(self):
        return self._price

    @staticmethod
    def _get_v(s2, k, v1, v2, corr):
        return math.sqrt(v1 ** 2 + (v2 * s2 / (s2 + k)) ** 2 - 2 * corr * v1 * v2 * s2 / (s2 + k))


# Implicit function for the price of a Barrier option
def _barrier_price_function(v, option_type, barrier_type, op, s0, k, t, r, q, h):
    return op - BarrierOption(option_type, barrier_type, s0, k, t, r, q, v, h).price


# Implicit function for the price of a Asian option
def _asian_price_function(v, option_type, op, s0, k, t, r, q):
    return op - AsianOption(option_type, s0, k, t, r, q, v).price


# Generic implied volatility function
def option_implied_vol(func, *args):
    logger.debug('Performing Newton-Raphson method')
    implied_vol, converged = optimize.newton(func=func,
                                             x0=0.25,
                                             args=args,
                                             full_output=True,
                                             disp=False)
    if converged.converged:
        logger.debug('Newton-Raphson method converged')
        return implied_vol

    logger.debug('Newton-Raphson method did not converged')
    logger.debug('Performing Bisection method')

    implied_vol, converged = optimize.bisect(f=func,
                                             a=0.005,
                                             b=2,
                                             args=args,
                                             full_output=True,
                                             disp=False)
    if converged.converged:
        logger.debug('Bisection method converged')
        return implied_vol
    else:
        raise RuntimeError(f'Bisection method dit not converged. Best guess: {implied_vol}')


# Implied volatility of a Barrier option on a asset that provides a yield at rate q
def barrier_implied_vol(option_type, barrier_type, op, s0, k, t, r, q, h):
    """
    :param option_type: 'c' or 'p'
    :param barrier_type: 'down-in', 'down-out', 'up-in' or 'up-out'
    :param op: option price
    :param s0: current asset price
    :param k: option strike price
    :param t: time to expiration of option in years
    :param r: continuously compounded risk-free rate of interest per annum
    :param q: continuously compounded yield rate provided by the asset
    :param h: barrier level
    :return: implied volatility of the option
    """
    return option_implied_vol(_barrier_price_function, option_type, barrier_type, op, s0, k, t, r, q, h)


# Implied volatility of a Asian option on a asset that provides a yield at rate q
def asian_implied_vol(option_type, op, s0, k, t, r, q):
    """
    :param option_type: 'c' or 'p'
    :param op: option price
    :param s0: current asset price
    :param k: option strike price
    :param t: time to expiration of option in years
    :param r: continuously compounded risk-free rate of interest per annum
    :param q: continuously compounded yield rate provided by the asset
    :return: implied volatility of the option
    """
    return option_implied_vol(_asian_price_function, option_type, op, s0, k, t, r, q)


if __name__ == '__main__':
    op = BarrierOption('c', 'down-out', 100, 110, 0.5, 0.01, 0.01, 0.2, 95)
    price = op.price
