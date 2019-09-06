import gamry_parser as parser
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class CyclicVoltammetry(parser.GamryParser):
    """Load a Cyclic Voltammetry experiment generated in Gamry EXPLAIN format."""

    def get_v_range(self):
        """ retrieve the programmed voltage scan ranges

        Args:
            None

        Returns:
            tuple, containing:
                float: voltage limit 1, in V
                float: voltage limit 2, in V

        """
        assert self.loaded, 'DTA file not loaded. Run CyclicVoltammetry.load()'
        assert 'VLIMIT1' in self.header.keys(), 'DTA header file missing VLIMIT1 specification'
        assert 'VLIMIT2' in self.header.keys(), 'DTA header file missing VLIMIT2 specification'

        return self.header['VLIMIT1'], self.header['VLIMIT2']

    def get_scan_rate(self):
        """ retrieve the programmed scan rate

        Args:
            None

        Returns:
            float: the scan rate, in mV/s

        """
        assert self.loaded, 'DTA file not loaded. Run CyclicVoltammetry.load()'
        assert 'SCANRATE' in self.header.keys(), 'DTA header file missing SCANRATE specification'
        return self.header['SCANRATE']

    def get_curve_data(self, curve=0):
        """ retrieve relevant cyclic voltammetry experimental data

        Args:
            curve (int, optional): curve number to return. Defaults to 0.

        Returns:
            pandas.DataFrame:
                - Vf: potential, in V
                - Im: current, in A

        """
        assert self.loaded, 'DTA file not loaded. Run CyclicVoltammetry.load()'
        assert curve >= 0, 'Invalid curve ({}). Indexing starts at 0'.format(curve)
        assert curve < self.curve_count, 'Invalid curve ({}). File contains {} total curves.'.format(curve, self.curve_count)
        df = self.curves[curve]
        
        return df[['Vf', 'Im']]

    def get_peaks(self, curve=0, minwidth=0.02, peak_range_x=None):
        """ retrieve peaks from cyclic voltammetry curve
        See scipy.signal.find_peaks for more details.
        Currently, the "peak" at the positive voltage limit is returned. This can be prevented with peak_range_x.
        The "peak" at the negative voltage limit is not returned.

        Args:
            curve (int, optional): curve number to find peaks. Defaults to 0.
            minwidth (float, optional): mininum peak width in volts. Defaults to 0.02 V = 20 mV
            peak_range_x: peaks outside of this range will be discarded. Two values required, e.g. [-1, 1]

        Returns:
            pandas.DataFrame:
                - peak_heights
                - prominences
                - left_bases
                - right_bases
                - widths
                - width_heights
                - left_ips
                - right_ips
                - peak_locations
                - curve

        """
        data = self.get_curve_data(curve)
        x = data['Vf'] # should be in V
        y = data['Im'] # should be in A
        
        dx = np.mean(np.abs(np.diff(x))) # V
        minwidth_elements = int(minwidth/dx)
        pks_pos, props_pos = find_peaks(y, width=minwidth_elements, height=0)
        pks_neg, props_neg = find_peaks(-y, width=minwidth_elements, height=0)
        props_neg['peak_heights'] = -props_neg['peak_heights']
        props_neg['prominences'] = -props_neg['prominences']
        pks = np.concatenate((pks_pos,pks_neg))
        props = dict()
        for key in props_pos.keys():
            props[key] = np.concatenate((props_pos[key], props_neg[key]))
        if peak_range_x is not None:
            outofrange = (x.iloc[pks]<peak_range_x[0]) | (x.iloc[pks]>peak_range_x[1])
            pks = np.delete(pks, np.where(outofrange))
            for key in props.keys():
                props[key] = np.delete(props[key], np.where(outofrange))
        props['peak_locations'] = x.iloc[pks]
        props = pd.DataFrame(props)
        props['curve'] = curve
        
        return props