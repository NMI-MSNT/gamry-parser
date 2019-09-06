import gamry_parser as parser
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib as mpl
import matplotlib.pyplot as plt
import quantities as pq


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

    def get_peaks(self, curve=0, width_V=0.02, peak_range_x=None):
        """ retrieve peaks from cyclic voltammetry curve
        See scipy.signal.find_peaks for more details. It would be useful to pass additional arguments to find_peaks.
        Currently, the "peak" at the positive voltage limit is returned. This can be prevented with peak_range_x.
        The "peak" at the negative voltage limit is not returned.

        Args:
            curve (int, optional): curve number to find peaks. Defaults to 0.
            width_V (float, optional): mininum peak width in volts. Defaults to 0.02 V = 20 mV
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
        width = int(width_V/dx)
        pks_pos, props_pos = find_peaks(y, width=width, height=0)
        pks_neg, props_neg = find_peaks(-y, width=width, height=0)
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

    def add_arrow(self, lines, number=None, position=None, size=15, color=None):
        """
        add an arrow to a line.

        line:       Line2D object
        number:     number of arrows, to be evenly distributed
        position:   x-position of the arrow. If None, mean of xdata is taken
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.
        
        Modified from
        https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
        """

        for line in lines:
            if color is None:
                color = line.get_color()
                
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            
            if number is not None:
                assert number>=1, 'Number must be larger than 1'
                assert position is None, 'Define either number or position, not both'
                
                n_ind = len(xdata)
                
                step = int(n_ind/(number+1))
                start_ind = 0
                n_arrows = 0
                for n in range(number):# start_ind<n_ind-step:
                    start_ind += step
                    end_ind = start_ind + 1
                    line.axes.annotate('',
                        xytext=(xdata[start_ind], ydata[start_ind]),
                        xy=(xdata[end_ind], ydata[end_ind]),
                        arrowprops=dict(arrowstyle="->", color=color),
                        size=size
                    )
                    n_arrows += 1
            else:
            
                if position is None:
                    position = xdata.mean()
                start_ind = np.argmin(np.abs(xdata - position))
                end_ind = start_ind + 1
            
                line.axes.annotate('',
                    xytext=(xdata[start_ind], ydata[start_ind]),
                    xy=(xdata[end_ind], ydata[end_ind]),
                    arrowprops=dict(arrowstyle="->", color=color),
                    size=size
                )
        
        
    def plot(self, ax=None, curves=None, units=None, peaks=True, peak_range_x=None, arrows=True, EOC=True, notes=True, cm='winter_r', verbose=False):
        """ plot cyclic voltammetry curve
        
        Args:
            ax:             matplotlib axis to plot on
            curves:         indices of curves to plot
            units:          units to plot. Default V, A
            peaks:          mark peaks?
            peak_range_x:   discard peaks outside of this range
            arrows:         show scan direction with arrows
            EOC:            show vertical line at EOC
            notes:          print notes on graph
            cm:             color map for curves. Default 'winter_r'
            verbose         print updates or not

        Returns:
            bool: successful or not

        """
        if ax is None:
            fig, ax = plt.subplots(1)
        
        x = 'Vf'
        y = 'Im'
        if units is None:
            units = {
                x: 'V',
                y: 'A'}
        stored_units = {
            x: 'V',
            y: 'A'}
        scaling = {
                x: float(pq.Quantity(1,stored_units[x]).rescale(units[x]).magnitude),
                y: float(pq.Quantity(1,stored_units[y]).rescale(units[y]).magnitude)
                }
            
        if self.get_experiment_type()!='CV':
            print(f'Cannot plot experiment type {self.get_experiment_type()} as CV ({self.fname})')
            return False
            
        vrange = self.get_v_range()
        if verbose:
            print(f'Experiment type: {self.get_experiment_type()}')
            print(f'Loaded curves: {self.get_curve_count()}')
            print(f'Programmed Scan Rate: {self.get_scan_rate():.0f} mV/s')
            print(f'Programmed V range: {vrange} V')#.format(vrange[0], vrange[1]))
            print('Notes:')
            for note in self.get_header()['NOTES'].split('\n'):
                print(f'\t{note}')

        all_curves = np.arange(self.get_curve_count())
        if curves is None:
            curves = all_curves
        else:
            if type(curves) is int:
                assert curves in all_curves, f'Cycle index {curves} invalid'
                curves = [curves,]
            else:
                for p in curves:
                    assert p in all_curves, f'Cycle index {p} invalid'
        # https://tonysyu.github.io/line-color-cycling.html
        n_lines = len(all_curves)
    #    cmap = plt.cm.winter_r
        cmap = plt.cm.get_cmap(cm,n_lines+1)
        c = np.arange(0., n_lines)
        norm = mpl.colors.BoundaryNorm(np.arange(len(c)+1)+0.5-1,len(c))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # this line may be ommitted for matplotlib >= 3.1
        
        color_idx = np.linspace(0, 1, n_lines)
        
        if EOC:
            ax.axvline(x=self.get_header()['EOC'],c='0.8')
        
        x = 'Vf'
        y = 'Im'
        
        for curve_idx in curves:
            curve = self.get_curve_data(curve_idx)
            if verbose:
                print(f'Plotting curve {curve_idx} with V range ({min(curve[x])}, {max(curve[y])})')
            
            line = ax.plot(curve[x]*scaling[x],curve[y]*scaling[y], color=cmap(color_idx[curve_idx])) #also user autumn
            if arrows:
                self.add_arrow(line, number=8)
            if peaks:
                curve_peaks = self.get_peaks(curve_idx, peak_range_x=peak_range_x)
                ax.plot(curve_peaks[curve_peaks['curve']==curve_idx]['peak_locations']*scaling[x],
                        curve_peaks[curve_peaks['curve']==curve_idx]['peak_heights']*scaling[y],
                        'rx')
            
        ax.set_xlabel(f'Voltage vs. ref ({units[x]})')
        ax.set_ylabel(f'Current ({units[y].replace("u","µ")})')

        plt.colorbar(sm, ticks=curves, orientation='vertical', label='Cycle', aspect=40)
        if notes:
            ax.text(0.02,0.98,self.get_header()['NOTES'],
                    verticalalignment='top',
                    horizontalalignment='left',
                    transform=ax.transAxes)
        
        return True