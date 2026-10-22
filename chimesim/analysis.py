"""Optional starter analyses of exported or in-memory simulation datasets."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from scipy.stats import rankdata
from chimesim import metrics
from chimesim.environment import Environment
from chimesim.configuration import flatten


def _pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError('Analysis requires pandas; install chimesim[analysis]') from exc
    return pd


def load(source):
    """Read a run folder or convert simulate() results to pandas tables."""
    pd = _pandas()
    if isinstance(source, (str, Path)):
        folder = Path(source).expanduser().resolve()
        data = {name: pd.read_csv(folder/(name+'.csv')) if (folder/(name+'.csv')).exists() else pd.DataFrame()
                for name in ('summary','repeats','individuals','trajectories')}
        config = json.loads((folder/'settings.json').read_text())
        data['settings'] = config.get('scenario', config)
        data['conditions'] = json.loads((folder/'conditions.json').read_text()) if (folder/'conditions.json').exists() else []
        return data
    if not isinstance(source, dict):
        raise TypeError('Provide simulate() results or a run folder')
    if 'output_dir' in source:
        return load(source['output_dir'])
    if 'output' in source and 'summary' not in source:
        return load(source['output'])
    data = dict(source)
    for name in ('summary','repeats','individuals','trajectories'):
        value = data.get(name)
        data[name] = value.copy() if isinstance(value,pd.DataFrame) else pd.DataFrame(value)
    return data


def _ranks(values):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.sum() > 1:
        result[valid] = (rankdata(values[valid], method='average')-1)/(valid.sum()-1)
    return result


def process(source, *, groups='largest', multiplier=2.5, link_distance=None):
    """Compute per-frame group metrics and mean individual positions.

    Physical groups include all agent types. 'largest' selects every tied largest
    component (minimum two agents); 'all' includes all such components. Group
    labels are local to a frame, not identities. Positional scores refer to the
    whole physical group; type-specific metrics additionally describe its types.
    """
    pd = _pandas()
    if groups not in {'largest','all'}:
        raise ValueError("groups must be 'largest' or 'all'")
    data = load(source)
    tracks, individuals = data['trajectories'], data['individuals']
    required = {'set','condition','rep','frame','t','id','x','y','vx','vy'}
    if not required.issubset(tracks):
        raise ValueError('Processing needs trajectories with positions and velocities; run with save_trajectories=True')
    if tracks.duplicated(['set','frame','id']).any():
        raise ValueError('Trajectory identities must be unique within each set/frame')
    if not {'set','id','agent_type'}.issubset(individuals):
        raise ValueError('Processing also needs the individuals table')
    if individuals.duplicated(['set','id']).any():
        raise ValueError('Individual identities must be unique within each set')
    joined = tracks.merge(individuals[['set','id','agent_type']], on=['set','id'], how='left', validate='many_to_one')
    if joined.agent_type.isna().any():
        raise ValueError('Every tracked individual needs an individuals row')
    scenarios = {c['condition']: c['scenario'] for c in data.get('conditions',[])}
    environments, labels = {}, {}
    for condition in joined.condition.unique():
        flat = flatten(scenarios.get(condition, data.get('settings',{})))
        if not {'width','height'}.issubset(flat):
            raise ValueError('Saved arena settings are needed for periodic-aware processing')
        environments[condition] = Environment(flat['width'],flat['height'],bounded=flat.get('bounded',True),
                                               mask=flat.get('mask'),wall_shapes=flat.get('wall_shapes',[]))
        values = next((c.get('values',{}) for c in data.get('conditions',[]) if c['condition']==condition),{})
        labels[condition] = ', '.join(f'{key}={value}' for key,value in values.items()) or f'Condition {condition}'
    group_rows, position_rows = [], []
    for set_id, recording in joined.groupby('set', sort=False):
        previous = {}
        for frame, snapshot in recording.groupby('frame',sort=True):
            condition, rep = int(snapshot.condition.iloc[0]), int(snapshot.rep.iloc[0])
            env = environments[condition]
            agents = [SimpleNamespace(unique_id=row.id,x=row.x,y=row.y,dx=row.vx,dy=row.vy) for row in snapshot.itertuples()]
            positions = metrics.within_group_positions(agents,env,multiplier,link_distance,largest_only=groups=='largest')
            components, sizes = np.unique(positions['component'][positions['component']>=0],return_counts=True)
            selected = components[sizes==sizes.max()] if len(sizes) and groups=='largest' else components
            types = snapshot.agent_type.to_numpy()
            current = {}
            for component in selected:
                indices = np.flatnonzero(positions['component']==component)
                base = dict(set=set_id, run=set_id, condition=condition, condition_label=labels[condition], rep=rep,
                            frame=int(frame), t=float(snapshot.t.iloc[0]), component=int(component))
                front = _ranks(positions['front_back'][indices])
                centre = _ranks(positions['centroid_distance'][indices])
                for i,f,c in zip(indices,front,centre):
                    position_rows.append(dict(base,id=agents[i].unique_id,agent_type=types[i],group_size=len(indices),
                                              front_rank=float(f),centre_rank=float(c),
                                              front_back=float(positions['front_back'][i]),
                                              centroid_distance=float(positions['centroid_distance'][i])))
                subsets = [(name,indices[types[indices]==name]) for name in sorted(set(types[indices]))]
                if len(subsets)>1:
                    subsets.insert(0,('All',indices))
                for name, subset in subsets:
                    state = metrics.group_state([agents[i] for i in subset],env)
                    key = (name,frozenset(agents[i].unique_id for i in subset))
                    centre_point = np.array([state['centroid_x'],state['centroid_y']])
                    centroid_speed = np.nan
                    if key in previous and np.isfinite(centre_point).all():
                        old_time, old_centre = previous[key]
                        delta = centre_point-old_centre
                        if not env.bounded:
                            size = np.array([env.width,env.height])
                            delta -= np.round(delta/size)*size
                        dt = base['t']-old_time
                        if dt>0:
                            centroid_speed=float(np.linalg.norm(delta)/dt)
                    current[key]=(base['t'],centre_point)
                    group_rows.append(dict(base,agent_type=name,centroid_speed=centroid_speed,
                                           spatial_groups=len(components),isolated=int((positions['component']<0).sum()),
                                           largest_group_fraction=float(sizes.max()/len(agents)) if len(sizes) else 0., **state))
            previous=current
    position_table=pd.DataFrame(position_rows)
    individual_summary=pd.DataFrame()
    structuring=pd.DataFrame()
    if len(position_table):
        individual_summary=position_table.groupby(['set','run','condition','condition_label','rep','id','agent_type'],as_index=False).agg(
            front_rank=('front_rank','mean'),centre_rank=('centre_rank','mean'),
            centroid_distance=('centroid_distance','mean'),position_samples=('frame','size'),
            front_samples=('front_rank','count'))
        trait_columns=[c for c in individuals if c not in individual_summary or c in {'set','id'}]
        individual_summary=individual_summary.merge(individuals[trait_columns],on=['set','id'],how='left',validate='one_to_one')
        structuring=individual_summary.groupby(['set','run','condition','condition_label','rep'],as_index=False).agg(
            front_rank_sd=('front_rank',lambda x: float(np.nanstd(x)) if x.notna().any() else np.nan),
            individuals=('id','size'),scored_individuals=('front_rank','count'))
    return dict(groups=pd.DataFrame(group_rows),positions=position_table,individuals=individual_summary,
                structuring=structuring,settings=data.get('settings'),conditions=data.get('conditions',[]))


def summary_table(source, table='summary'):
    """Convert existing simulator summary columns to rows per agent type."""
    pd = _pandas()
    data = load(source)
    frame = data[table]
    if frame.empty:
        return frame
    names = list(data.get('settings',{}).get('agents',{}))
    if not names:
        suffix='_n_mean' if table=='repeats' else '_n'
        names = [c[:-len(suffix)] for c in frame if c.endswith(suffix)]
    result=[]
    metric_suffix='_mean' if table=='repeats' else ''
    for name in names:
        columns={f'{name}_{key}{metric_suffix}':key for key in ('n','polarisation','nnd','speed')}
        kept=[c for c in frame if not any(c.startswith(n+'_') for n in names) and c!='overall_polarisation'+metric_suffix]
        item=frame[kept+[c for c in columns if c in frame]].rename(columns=columns)
        item['agent_type']=name
        if 'condition' in item:
            labels={c['condition']: ', '.join(f'{k}={v}' for k,v in c.get('values',{}).items()) for c in data.get('conditions',[])}
            item['condition_label']=item['condition'].map(lambda c: labels.get(c) or f'Condition {c}')
        if 'set' in item:
            item['run']=item['set']
        result.append(item)
    return pd.concat(result,ignore_index=True) if result else pd.DataFrame()
