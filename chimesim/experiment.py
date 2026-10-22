"""Streaming experiment output and a separate no-code batch-runner entry point."""
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from chimesim.simulate import simulate, _resolve_settings, _condition_grid
from chimesim.configuration import scenario_from_flat, _RUN


def run_experiment(settings, output_directory, *, progress=None, should_cancel=None, conditions=None, performance=False):
    """Write a new run folder; cancellation/errors preserve all written rows."""
    if conditions is None:
        from chimesim.configuration import load_file
        supplied = load_file(settings) if isinstance(settings, (str, Path)) else settings
        conditions = supplied.get('conditions')
    resolved, _ = _resolve_settings(settings=settings)
    if resolved['seed'] is None:
        import secrets
        resolved['seed'] = secrets.randbits(32)
    grid = _condition_grid(scenario_from_flat(resolved), conditions)
    parent = Path(output_directory).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')
    folder.mkdir()
    configuration = dict(scenario=scenario_from_flat(resolved), run={k: resolved[k] for k in _RUN})
    if conditions:
        configuration['conditions'] = conditions
        configurations = [dict(condition=i, values=values, scenario=scenario,
                               run=dict(configuration['run'], seed=resolved['seed'] + i * resolved['n_reps']))
                          for i, (scenario, values) in enumerate(grid)]
        (folder / 'conditions.json').write_text(json.dumps(configurations, indent=2, allow_nan=False), encoding='utf-8')
    (folder / 'settings.json').write_text(json.dumps(configuration, indent=2, allow_nan=False), encoding='utf-8')
    data_format = dict(version=1, identity=['set', 'id'], set_scope='Unique condition/repeat within this run folder; prefix sets when combining run folders.',
                       trajectory_columns=['set', 'condition', 'rep', 'frame', 't', 'id', 'x', 'y', 'vx', 'vy', 'speed', 'heading_rad'],
                       time='frame and t count completed model steps, including burn-in; first recorded frame is burn_in + 1. One t unit is one model step, not a measured second.',
                       position='x/y are model coordinates (pixels); y increases downwards in the GUI. Periodic arenas store wrapped positions.',
                       velocity='vx/vy are model velocities per step; speed is their magnitude, not a difference of wrapped positions.',
                       heading='heading_rad is atan2(vy, vx), modulo 2*pi, using retained heading when stationary.',
                       individuals='One row per initialised agent before burn-in, including fixed traits and actual sampled assignments. Trait angles are radians.',
                       repeats='Arithmetic time means of recorded per-step metrics; excludes burn-in. cancelled and recorded_steps identify partial repeats. Missing means indicate no recorded steps.')
    (folder / 'data_format.json').write_text(json.dumps(data_format, indent=2), encoding='utf-8')
    started = time.perf_counter()
    metadata = dict(status='running', completed_steps=0, completed_reps=0)
    def write_status():
        metadata['elapsed_seconds'] = time.perf_counter() - started
        (folder / 'status.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    write_status()
    handles, writers = {}, {}
    def row(kind, values):
        if kind not in writers:
            fields = list(values)
            if kind == 'repeats':
                from chimesim.metrics import compute
                metric_names = compute({name: [] for name in resolved['agents']})
                fields = ['set', 'condition', 'rep', 'seed', 'recorded_steps', 'cancelled'] + [name + '_mean' for name in metric_names]
            handles[kind] = (folder / (kind + '.csv')).open('w', newline='', encoding='utf-8')
            writers[kind] = csv.DictWriter(handles[kind], fieldnames=fields)
            writers[kind].writeheader()
        writers[kind].writerow(values)
    def report(update):
        metadata.update(completed_steps=update['completed'], repeat=update['repeat'], phase=update['phase'])
        if update['phase'] == 'Repeat complete':
            metadata['completed_reps'] = update.get('condition', 0) * resolved['n_reps'] + update['repeat']
        if progress:
            progress({**update, 'output': str(folder)})
    try:
        result = simulate(settings=resolved, conditions=conditions, progress=report, should_cancel=should_cancel,
                          on_row=row, collect_results=False, verbose=False, performance=performance)
        if performance:
            metadata['performance'] = result['performance']
            (folder / 'performance.json').write_text(json.dumps(result['performance'], indent=2, allow_nan=False), encoding='utf-8')
        metadata.update(status='cancelled' if result['cancelled'] else 'complete',
                        completed_steps=result['completed_steps'], completed_reps=result['completed_reps'])
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        for handle in handles.values():
            handle.close()
        write_status()
    return dict(output_dir=str(folder), output=str(folder), **metadata)


def launch(settings=None, conditions=None, performance=False):
    """Open the standalone experiment runner; import Tk only when requested."""
    resolved, _ = _resolve_settings(settings=settings)
    if len(resolved['agents']) > 3:
        raise ValueError('The simulation setup window supports up to three agent types')
    try:
        from chimesim.ui_simulation import SimulationSetup
    except ModuleNotFoundError as exc:
        if exc.name not in {'tkinter', '_tkinter'}:
            raise
        raise RuntimeError('The experiment runner requires Python with Tkinter installed') from exc
    app = SimulationSetup(settings=resolved, conditions=conditions, **({"performance": True} if performance else {}))
    app.mainloop()
    return app.result


def main():
    """Console entry point: saved-run dictionaries are Python return values."""
    launch()


if __name__ == '__main__':
    main()
