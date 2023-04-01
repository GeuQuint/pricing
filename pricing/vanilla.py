import math
from scipy.stats import norm
from scipy import optimize
import logging

# Set logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d: %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


# Pricing for a European option on a asset that provides a yield at rate q
class OptionPricing:
    def __init__(self, option_type, s0, k, t, r, q, v):
        """
        :param option_type: 'c' or 'p'
        :param s0: current asset price
        :param k: option strike price
        :param t: time to expiration of option
        :param r: continuously compounded risk-free rate of interest per annum
        :param q: continuously compounded yield rate provided by the asset
        :param v: implied volatility of the option
        """
        self.option_type = option_type
        self.s0 = s0
        self.k = k
        self.r = r
        self.q = q
        self.t = t
        self.v = v
        self._d1 = (math.log(s0 / k) + (r - q + v ** 2 / 2) * t) / (v * math.sqrt(t))
        self._d2 = self._d1 - v * math.sqrt(t)
        self._price = self._get_price(option_type, s0, k, t, r, q, self._d1, self._d2)
        self._delta, self._gamma, self._theta, self._vega, self._rho = \
            self._get_greek_letters(option_type, s0, k, t, r, q, v, self._d1, self._d2)

    def __repr__(self):
        return f"OptionPricing(option_type='{self.option_type}', s0={self.s0}, k={self.k}, t={self.t}, " \
               f"r={self.r}, q={self.q}, v={self.v})"

    @property
    def price(self):
        return self._price

    @property
    def delta(self):
        return self._delta

    @property
    def gamma(self):
        return self._gamma

    @property
    def theta(self):
        return self._theta

    @property
    def vega(self):
        return self._vega

    @property
    def rho(self):
        return self._rho

    @staticmethod
    def _get_price(option_type, s0, k, t, r, q, d1, d2):
        if option_type == 'c':
            return s0 * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
        if option_type == 'p':
            return k * math.exp(-r * t) * norm.cdf(-d2) - s0 * math.exp(-q * t) * norm.cdf(-d1)
        return None

    @staticmethod
    def _get_greek_letters(option_type, s0, k, t, r, q, v, d1, d2):
        if option_type == 'c':
            delta = math.exp(-q * t) * norm.cdf(d1)
            gamma = norm.pdf(d1) * math.exp(-q * t) / (s0 * v * math.sqrt(t))
            theta = -s0 * norm.pdf(d1) * v * math.exp(-q * t) / (2 * math.sqrt(t)) + \
                q * s0 * norm.cdf(d1) * math.exp(-q * t) - r * k * math.exp(-r * t) * norm.cdf(d2)
            vega = s0 * math.sqrt(t) * norm.pdf(d1) * math.exp(-q * t)
            rho = k * t * math.exp(-r * t) * norm.cdf(d2)
            return delta, gamma, theta, vega, rho

        elif option_type == 'p':
            delta = math.exp(-q * t) * (norm.cdf(d1) - 1)
            gamma = norm.pdf(d1) * math.exp(-q * t) / (s0 * v * math.sqrt(t))
            theta = -s0 * norm.pdf(d1) * v * math.exp(-q * t) / (2 * math.sqrt(t)) - \
                q * s0 * norm.cdf(-d1) * math.exp(-q * t) + r * k * math.exp(-r * t) * norm.cdf(-d2)
            vega = s0 * math.sqrt(t) * norm.pdf(d1) * math.exp(-q * t)
            rho = -k * t * math.exp(-r * t) * norm.cdf(-d2)
            return delta, gamma, theta, vega, rho

        return None


# Pricing for a European option on a currency
class CurrencyOption(OptionPricing):
    def __init__(self, option_type, s0, k, t, r, rf, v):
        """
        :param option_type: 'c' or 'p'
        :param s0: spot exchange rate (the value of one unit of the foreign currency)
        :param k: option strike price
        :param t: time to expiration of option
        :param r: continuously compounded risk-free rate of interest per annum
        :param rf: foreign risk-free rate of interest
        :param v: implied volatility of the option
        """
        super().__init__(option_type, s0, k, t, r, rf, v)
        self.rf = rf
        self.rho_foreign = self._get_rho_foreign(option_type, s0, t, rf, self._d1)

    def __repr__(self):
        return f"CurrencyOption(option_type='{self.option_type}', s0={self.s0}, k={self.k}, t={self.t}, r={self.r}, " \
               f"rf={self.rf}, v={self.v})"

    @staticmethod
    def _get_rho_foreign(option_type, s0, t, rf, d1):
        if option_type == 'c':
            return -t * math.exp(-rf * t) * s0 * norm.cdf(d1)
        elif option_type == 'p':
            return t * math.exp(-rf * t) * s0 * norm.cdf(-d1)
        return None


# Pricing for a European option on a futures contract
class FuturesOption(OptionPricing):
    def __init__(self, option_type, f0, k, t, r, v):
        """
        :param option_type: 'c' or 'p'
        :param f0: current futures price
        :param k: option strike price
        :param t: time to expiration of option
        :param r: continuously compounded risk-free rate of interest per annum
        :param v: implied volatility of the option
        """
        self.f0 = f0
        super().__init__(option_type, f0, k, t, r, r, v)

    def __repr__(self):
        return f"FuturesOption(option_type='{self.option_type}', f0={self.f0}, k={self.k}, t={self.t}, r={self.r}, " \
               f"v={self.v})"


# Implicit function for the price of a European option
def _price_function(v, option_type, op, s0, k, t, r, q):
    return op - OptionPricing(option_type, s0, k, t, r, q, v).price


# Implied volatility of a European option on a asset that provides a yield at rate q
def option_implied_vol(option_type, op, s0, k, t, r, q):
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

    logger.debug('Performing Newton-Raphson method')
    implied_vol, converged = optimize.newton(func=_price_function,
                                             x0=0.25,
                                             args=(option_type, op, s0, k, t, r, q),
                                             full_output=True,
                                             disp=False)
    if converged.converged:
        logger.debug('Newton-Raphson method converged')
        return implied_vol

    logger.debug('Newton-Raphson method did not converged')

    logger.debug('Performing Bisection method')

    implied_vol, converged = optimize.bisect(f=_price_function,
                                             a=0.005,
                                             b=2,
                                             args=(option_type, op, s0, k, t, r, q),
                                             full_output=True,
                                             disp=False)
    if converged.converged:
        logger.debug('Bisection method converged')
        return implied_vol
    else:
        raise RuntimeError(f'Bisection method dit not converged. Best guess: {implied_vol}')


# Implied volatility of a European option on a currency
def currency_implied_vol(option_type, op, s0, k, t, r, rf):
    """
    :param option_type: 'c' or 'p'
    :param op: option price
    :param s0: spot exchange rate (the value of one unit of the foreign currency)
    :param k: option strike price
    :param t: time to expiration of option in years
    :param r: continuously compounded risk-free rate of interest per annum
    :param rf: foreign risk-free rate of interest
    :return: implied volatility of the currency option
    """
    return option_implied_vol(option_type, op, s0, k, t, r, rf)


# Implied volatility of a European option on a futures contract
def futures_implied_vol(option_type, op, f0, k, t, r):
    """
    :param option_type: 'c' or 'p'
    :param op: option price
    :param f0: current futures price
    :param k: option strike price
    :param t: time to expiration of option in years
    :param r: continuously compounded risk-free rate of interest per annum
    :return: implied volatility of the futures option
    """
    return option_implied_vol(option_type, op, f0, k, t, r, r)


if __name__ == '__main__':
    option_type = 'c'
    s0 = 0.6605
    k = 0.681
    t = 21/360
    r = 0.1
    rf = 0.09
    v = 0.135

    option = CurrencyOption(option_type, s0, k, t, r, rf, v)
    price, delta, gamma, theta, vega = option.price, option.delta, option.gamma, option.theta, option.vega
