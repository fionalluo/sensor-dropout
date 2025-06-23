import time
import random 

class Every:

  def __init__(self, every, initial=True):
    self._every = every
    self._initial = initial
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    if self._prev is None:
      self._prev = (step // self._every) * self._every
      return self._initial
    if step >= self._prev + self._every:
      self._prev += self._every
      return True
    return False


class EveryAnneal:
    """This class is used to anneal the frequency of an event happening over time."""

    def __init__(self, start_freq, end_freq, start_anneal, end_anneal, steps):
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.start_step = int(start_anneal * steps)
        self.end_step = int(end_anneal * steps)
        self.steps = steps  # Stored for reference, not used in calculation

    def __call__(self, step):
        step = int(step)
        if step < self.start_step:
            prob = self.start_freq
        elif step >= self.end_step:
            prob = self.end_freq
        else:
            phase_length = self.end_step - self.start_step
            fraction = (step - self.start_step) / phase_length
            prob = self.start_freq + (self.end_freq - self.start_freq) * fraction
        return random.random() < prob

class Anneal:
  """This class returns an annealed frequency over time. (Not discrete/binary, like EveryAnneal)"""
  def __init__(self, start_freq: float, end_freq: float, start_anneal: float, end_anneal: float, steps: int) -> None:
    self.start_freq = start_freq
    self.end_freq = end_freq
    self.start_step = int(start_anneal * steps)
    self.end_step = int(end_anneal * steps)
    self.steps = steps  # Stored for reference, not used in calculation
    self.step = 0
    self.prob = start_freq

  def step(self) -> None:
    self.step += 1
    if self.step < self.start_step:
        prob = self.start_freq
    elif self.step >= self.end_step:
        prob = self.end_freq
    else:
        phase_length = self.end_step - self.start_step
        fraction = (self.step - self.start_step) / phase_length
        prob = self.start_freq + (self.end_freq - self.start_freq) * fraction
    self.prob = prob  # return the annealed PROBABILITY
  
  def get_prob(self) -> float:
    return self.prob


class Ratio:

  def __init__(self, ratio):
    assert ratio >= 0, ratio
    self._ratio = ratio
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    step = int(step)
    if not self._until:
      return True
    return step < self._until


class Clock:

  def __init__(self, every):
    self._every = every
    self._prev = None

  def __call__(self, step=None):
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    now = time.time()
    if self._prev is None:
      self._prev = now
      return True
    if now >= self._prev + self._every:
      # self._prev += self._every
      self._prev = now
      return True
    return False
