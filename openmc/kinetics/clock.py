
import copy
import numpy as np

TIME_POINTS = ['START',
               'PREVIOUS_OUTER',
               'FORWARD_OUTER',
               'PREVIOUS_INNER',
               'FORWARD_INNER',
               'END']


class Clock(object):

    def __init__(self, start=0., end=3., dt_outer=1.e-1, dt_inner=1.e-2):

        # Initialize coordinates
        self.dt_outer = dt_outer
        self.dt_inner = dt_inner

        # Create a dictionary of clock times
        self._times = {}
        for t in TIME_POINTS:
            self._times[t] = start

        # Reset the end time
        self._times['END'] = end

    def __repr__(self):

        string = 'Clock\n'
        string += '{0: <24}{1}{2}\n'.format('\tdt inner', '=\t', self.dt_inner)
        string += '{0: <24}{1}{2}\n'.format('\tdt outer', '=\t', self.dt_outer)

        for t in TIME_POINTS:
            string += '{0: <24}{1}{2}\n'.format('\tTime ' + t, '=\t', self.times[t])

        return string

    @property
    def dt_inner(self):
        return self._dt_inner

    @property
    def dt_outer(self):
        return self._dt_outer

    @property
    def times(self):
        return self._times

    @dt_inner.setter
    def dt_inner(self, dt_inner):
        self._dt_inner = np.float64(dt_inner)

    @dt_outer.setter
    def dt_outer(self, dt_outer):
        self._dt_outer = np.float64(dt_outer)

    @times.setter
    def times(self, times):
        self._times = np.float64(times)
