"""Small optional plotting interface; figures remain editable Matplotlib axes."""
from numbers import Real
import numpy as np
from scipy.stats import gaussian_kde
from chimesim.analysis import load, summary_table, _pandas


def plot(source, variable, *, kind='time', category='agent_type', table=None,
         unit='repeat', x=None, ax=None, time_bin=None):
    """Plot time series, boxes, densities or scatter from result/analysis tables.

    time_bin optionally averages time-series observations in fixed-width bins
    of the plotted time coordinate, independently within each repeat.
    category accepts one column or a list, e.g. ['condition', 'agent_type'].
    Boxes/densities use one mean per repeat by default; unit='individual' uses
    individual means and unit='frame' deliberately pools frame observations.
    Use table='individuals' for mean positions and traits, table='positions' for
    positional histories, or table='groups' for processed physical-group metrics.
    """
    if time_bin is not None and (kind != 'time' or isinstance(time_bin, bool) or
                                 not isinstance(time_bin, Real) or not np.isfinite(time_bin) or time_bin <= 0):
        raise ValueError('time_bin must be a positive finite number for a time plot')
    pd = _pandas()
    if kind not in {'time','box','density','scatter'}:
        raise ValueError('kind must be time, box, density or scatter')
    if unit not in {'repeat','individual','frame'}:
        raise ValueError('unit must be repeat, individual or frame')
    if isinstance(source,pd.DataFrame):
        frame=source.copy()
    elif isinstance(source,dict) and 'groups' in source:
        selected = table or ('individuals' if variable in {'front_rank','centre_rank'} and kind!='time' else 'positions' if variable in {'front_rank','centre_rank'} else 'groups')
        frame=source[selected].copy()
    else:
        data=load(source)
        if table in {'individuals','trajectories'}:
            frame=data[table].copy()
        else:
            frame=summary_table(data, table or 'summary')
    if variable not in frame:
        raise ValueError(f'No {variable!r} column; choose a matching table or process saved trajectories first')
    categories = [category] if isinstance(category,str) else list(category or [])
    categories = [{'type':'agent_type'}.get(c,c) for c in categories]
    categories = ['condition_label' if c=='condition' and 'condition_label' in frame else c for c in categories]
    if any(c not in frame for c in categories):
        raise ValueError(f'Unknown category; available columns: {list(frame.columns)}')
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError('Plotting requires matplotlib; install chimesim[analysis]') from exc
    if ax is None:
        _,ax=plt.subplots()
    frame=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=[variable])
    if frame.empty:
        raise ValueError('No finite observations to plot')
    if kind in {'box','density'} and unit!='frame':
        keys=list(dict.fromkeys(categories+(['set'] if 'set' in frame else [])))
        if unit=='individual':
            if 'id' not in frame:
                raise ValueError("unit='individual' needs an id column")
            keys=list(dict.fromkeys(keys+['id']))
        if not keys:
            raise ValueError("Repeat aggregation needs a set column; use unit='frame' for ungrouped observations")
        frame=frame.groupby(keys,dropna=False,as_index=False)[variable].mean()
    def grouped(data, keys):
        if not keys:
            return [((),data)]
        return data.groupby(keys[0] if len(keys)==1 else keys,sort=False,dropna=False)
    def label(key):
        values=key if isinstance(key,tuple) else (key,)
        return ' · '.join(str(v) for v in values) or variable.replace('_',' ')
    if kind=='time':
        axis=x or ('t' if 't' in frame else 'frame' if 'frame' in frame else 'step')
        if axis not in frame:
            raise ValueError('Time series need a t, frame or step column')
        seen=set()
        for key,data in grouped(frame,categories):
            colour=ax._get_lines.get_next_color()
            for _,recording in grouped(data,['set'] if 'set' in data else []):
                recording=recording.groupby(axis,as_index=False)[variable].mean().sort_values(axis)
                if time_bin is not None:
                    bins = np.floor(recording[axis] / time_bin)
                    recording = recording.groupby(bins.to_numpy())[[axis, variable]].mean().reset_index(drop=True)
                text=label(key)
                ax.plot(recording[axis],recording[variable],color=colour,alpha=.65,linewidth=.8,
                        label=text if text not in seen else None)
                seen.add(text)
        ax.set_xlabel(axis.replace('_',' '))
        ax.set_ylabel(variable.replace('_',' '))
    elif kind=='box':
        parts=list(grouped(frame,categories))
        ax.boxplot([part[variable].to_numpy() for _,part in parts])
        ax.set_xticks(np.arange(1,len(parts)+1))
        ax.set_xticklabels([label(key) for key,_ in parts],rotation=25,ha='right')
        ax.set_ylabel(variable.replace('_',' '))
    elif kind=='density':
        for key,data in grouped(frame,categories):
            values=data[variable].to_numpy(dtype=float)
            if len(values)<2 or np.ptp(values)<1e-12:
                ax.axvline(values.mean(),label=label(key))
            else:
                extent=np.ptp(values)*.15
                axis=np.linspace(values.min()-extent,values.max()+extent,256)
                ax.plot(axis,gaussian_kde(values)(axis),label=label(key))
        ax.set_xlabel(variable.replace('_',' '))
        ax.set_ylabel('Density')
    else:
        if x is None or x not in frame:
            raise ValueError('Scatter plots need x= with a valid column')
        for key,data in grouped(frame,categories):
            ax.scatter(data[x],data[variable],alpha=.65,label=label(key))
        ax.set_xlabel(x.replace('_',' '))
        ax.set_ylabel(variable.replace('_',' '))
    if categories and kind!='box':
        ax.legend(frameon=False)
    return ax
