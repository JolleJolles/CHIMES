"""Standalone configuration and progress window, separate from the live GUI."""
import copy
import gc
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from chimesim.agents import BOID
from chimesim.simulate import _load_file, _resolve_settings
from chimesim.experiment import run_experiment
from chimesim.configuration import scenario_from_flat, _RUN
from chimesim.simulate import _condition_grid


class SimulationSetup(tk.Tk):
    def __init__(self, settings=None, conditions=None, performance=False):
        super().__init__()
        self.title('CHIMES — Simulation setup')
        self.geometry('760x680')
        self.minsize(650, 580)
        self.loaded = {}
        self.agent_templates = []
        self.agent_vars = []
        self.agent_original_names = []
        self.relationship_vars = {}
        self.condition_vars = []
        self.cancel_event = threading.Event()
        self.messages = queue.Queue()
        self.worker = None
        self.result = None
        self.closing = False
        self.controls = []
        self.values = {}
        self.source = tk.StringVar(value='Default configuration')
        top = ttk.Frame(self, padding=10)
        top.pack(fill='both', expand=True)
        files = ttk.Frame(top); files.pack(fill='x')
        ttk.Button(files, text='Load settings', command=self.load).pack(side='left')
        ttk.Button(files, text='Save settings', command=self.save).pack(side='left', padx=5)
        ttk.Label(files, textvariable=self.source).pack(side='left', padx=8)
        self.tabs = ttk.Notebook(top); self.tabs.pack(fill='both', expand=True, pady=10)
        self.experiment, self.agents, self.environment = [ttk.Frame(self.tabs, padding=12) for _ in range(3)]
        for frame, title in ((self.experiment, 'Experiment'), (self.agents, 'Agents'), (self.environment, 'Arena and movement')):
            self.tabs.add(frame, text=title)
        for key, title, value in [('n_steps', 'Recorded steps per repeat', 1000), ('burn_in', 'Burn-in steps', 200),
                                   ('n_reps', 'Repeats', 5), ('seed', 'Seed (blank = random)', '')]:
            self.field(self.experiment, key, title, value)
        self.record = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.experiment, text='Save agent trajectories', variable=self.record).pack(anchor='w', pady=6)
        self.performance = tk.BooleanVar(value=performance)
        ttk.Checkbutton(self.experiment, text='Save performance report', variable=self.performance).pack(anchor='w', pady=4)
        self.field(self.experiment, 'output', 'Output folder', str(Path.home() / 'CHIMESim-results'))
        ttk.Button(self.experiment, text='Choose output folder', command=self.choose_output).pack(anchor='w')
        ttk.Label(self.experiment, text='Each run gets a new folder. Stop simulation keeps data already written.', wraplength=600).pack(anchor='w', pady=12)
        tools = ttk.Frame(self.agents); tools.pack(fill='x')
        ttk.Button(tools, text='Add agent type', command=self.add_agent).pack(side='left')
        ttk.Button(tools, text='Remove last type', command=self.remove_agent).pack(side='left', padx=5)
        self.agent_body = ttk.Frame(self.agents); self.agent_body.pack(fill='x', pady=8)
        self.relationship_body = ttk.Frame(self.agents); self.relationship_body.pack(fill='x', pady=8)
        ttk.Label(self.agents, text='Loaded distributions and advanced traits are retained. Distributions override the type defaults shown here.',
                  wraplength=650).pack(anchor='w', pady=6)
        for key, title, value in [('width', 'Arena width', 1200), ('height', 'Arena height', 800),
                                  ('desired_speed', 'Movement speed', 3), ('fov_deg', 'Field of view (degrees)', 270),
                                  ('repulsion_fov_deg', 'Repulsion FOV (blank = same)', ''),
                                  ('visual_range', 'Visual range (blank = unlimited)', 200),
                                  ('wall_avoid_radius', 'Wall avoidance distance', 50), ('wall_strength', 'Wall strength', 6),
                                  ('capture_distance', 'Capture distance (requires Captures)', 10), ('thigmotaxis', 'Wall following', 0), ('mask', 'Wall mask image (optional)', '')]:
            self.field(self.environment, key, title, value)
        self.bounded = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.environment, text='Bounded arena (otherwise periodic)', variable=self.bounded).pack(anchor='w')
        model_row = ttk.Frame(self.environment); model_row.pack(fill='x', pady=3)
        ttk.Label(model_row, text='Interaction model', width=30).pack(side='left')
        self.model = tk.StringVar(value='zonal')
        ttk.Combobox(model_row, textvariable=self.model, values=['zonal','voronoi','From settings'], state='readonly', width=20).pack(side='left')
        ttk.Button(self.environment, text='Choose mask image', command=self.choose_mask).pack(anchor='w')
        self.conditions_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.conditions_tab, text='Conditions')
        ttk.Label(self.conditions_tab, text='Vary parameters across all combinations; repeats run for each combination.',
                  wraplength=640).pack(anchor='w', pady=6)
        ttk.Label(self.conditions_tab, text='Choose a parameter, then enter its values, e.g. 0.5, 1, 1.5. JSON lists also work.',
                  wraplength=640).pack(anchor='w', pady=6)
        self.condition_body = ttk.Frame(self.conditions_tab)
        self.condition_body.pack(fill='x')
        ttk.Button(self.conditions_tab, text='Add parameter', command=self.add_condition).pack(anchor='w', pady=6)
        self.condition_count = tk.StringVar(value='One condition')
        ttk.Label(self.conditions_tab, textvariable=self.condition_count).pack(anchor='w')
        ttk.Button(self.conditions_tab, text='Check combinations', command=self.check_conditions).pack(anchor='w', pady=6)
        self.status = tk.StringVar(value='Ready')
        ttk.Label(top, textvariable=self.status, wraplength=700).pack(fill='x', pady=5)
        self.bar = ttk.Progressbar(top, maximum=100); self.bar.pack(fill='x')
        self.timing = tk.StringVar(value='')
        ttk.Label(top, textvariable=self.timing).pack(anchor='w', pady=4)
        buttons = ttk.Frame(top); buttons.pack(fill='x', pady=5)
        self.start_button = ttk.Button(buttons, text='Simulate', command=self.start); self.start_button.pack(side='left')
        self.cancel_button = ttk.Button(buttons, text='Stop simulation', command=self.cancel, state='disabled'); self.cancel_button.pack(side='left', padx=5)
        self.exit_button = ttk.Button(buttons, text='Exit', command=self.close); self.exit_button.pack(side='right')
        self.protocol('WM_DELETE_WINDOW', self.close)
        if settings is None:
            self.add_agent()
        else:
            self.apply_settings(settings)
        for path, values in (conditions or {}).items():
            self.add_condition(path, values)
        self.after(100, self.poll)

    def destroy(self):
        # Remove polling callbacks before a subsequent Tk session is opened.
        for callback in self.tk.call('after', 'info'):
            self.after_cancel(callback)
        super().destroy()

    def field(self, parent, key, label, value):
        row = ttk.Frame(parent); row.pack(fill='x', pady=3)
        ttk.Label(row, text=label, width=32).pack(side='left')
        var = tk.StringVar(value=str(value)); self.values[key] = var
        ttk.Entry(row, textvariable=var).pack(side='left', fill='x', expand=True)

    def add_agent(self, template=None):
        if len(self.agent_vars) >= 3:
            return
        template = copy.deepcopy(template or BOID.clone(f'A{len(self.agent_vars)+1}').to_dict())
        name = template.pop('_name', f'A{len(self.agent_vars)+1}')
        self.agent_templates.append(template)
        self.agent_original_names.append(name)
        self.agent_vars.append({key: tk.StringVar(value=str(value)) for key,value in
                                dict(name=name, n_default=template.get('n_default',100), size_multiplier=template.get('size_multiplier',1), speed_scale=template.get('speed_scale',1),
                                     r_r=template.get('r_r',10), r_o=template.get('r_o',30), r_a=template.get('r_a',75)).items()})
        self.render_agents()

    def remove_agent(self):
        if len(self.agent_vars) > 1:
            self.agent_vars.pop(); self.agent_templates.pop(); self.agent_original_names.pop(); self.render_agents()

    def render_agents(self):
        for child in self.agent_body.winfo_children(): child.destroy()
        keys = ['name','n_default','size_multiplier','speed_scale','r_r','r_o','r_a']
        for col,label in enumerate(['Type','Agents','Body size ×','Speed ×','Repulsion','Alignment','Attraction']):
            ttk.Label(self.agent_body,text=label).grid(row=0,column=col,padx=4)
        for row,variables in enumerate(self.agent_vars,1):
            for col,key in enumerate(keys):
                ttk.Entry(self.agent_body,textvariable=variables[key],width=10).grid(row=row,column=col,padx=4,pady=5)

        self.render_relationships()

    def render_relationships(self):
        for child in self.relationship_body.winfo_children():
            child.destroy()
        previous = self.relationship_vars
        self.relationship_vars = {}
        if len(self.agent_vars) < 2:
            return
        ttk.Label(self.relationship_body, text='Directed relationships — capture is off unless selected').grid(row=0, column=0, columnspan=4, sticky='w')
        row = 1
        for i, source in enumerate(self.agent_vars):
            for j, target in enumerate(self.agent_vars):
                if i == j:
                    continue
                label = ttk.Frame(self.relationship_body)
                label.grid(row=row, column=0, sticky='w', padx=4)
                ttk.Label(label, textvariable=source['name']).pack(side='left')
                ttk.Label(label, text=' → ').pack(side='left')
                ttk.Label(label, textvariable=target['name']).pack(side='left')
                values = previous.get((i,j)) or {key: tk.BooleanVar(value=self.agent_original_names[j] in self.agent_templates[i].get(key, []))
                                                for key in ('pursues','flees','captures')}
                self.relationship_vars[i,j] = values
                def change(key, values=values):
                    opposite = {'pursues':'flees', 'flees':'pursues'}.get(key)
                    if opposite and values[key].get():
                        values[opposite].set(False)
                for col, key in enumerate(('pursues','flees','captures'),1):
                    ttk.Checkbutton(self.relationship_body, text=key.capitalize(), variable=values[key],
                                    command=lambda key=key, change=change: change(key)).grid(row=row, column=col, padx=5)
                row += 1

    def configuration(self):
        config = copy.deepcopy(self.loaded)
        for key in ['n_steps','burn_in','n_reps','width','height']:
            config[key] = int(self.values[key].get())
        for key in ['desired_speed','fov_deg','wall_avoid_radius','wall_strength','thigmotaxis','capture_distance']:
            config[key] = float(self.values[key].get())
        config['repulsion_fov_deg'] = float(self.values['repulsion_fov_deg'].get()) if self.values['repulsion_fov_deg'].get().strip() else None
        config['visual_range'] = float(self.values['visual_range'].get()) if self.values['visual_range'].get().strip() else None
        config['seed'] = int(self.values['seed'].get()) if self.values['seed'].get().strip() else None
        config.update(bounded=self.bounded.get(),save_trajectories=self.record.get(),mask=self.values['mask'].get().strip() or None)
        if self.model.get() != 'From settings': config['interaction_model'] = self.model.get()
        else: config.pop('interaction_model',None)
        agents = {}
        for template,variables in zip(self.agent_templates,self.agent_vars):
            name = variables['name'].get().strip()
            if not name or name in agents: raise ValueError('Agent types need unique nonempty names')
            values = copy.deepcopy(template)
            values['n_default'] = int(variables['n_default'].get())
            for key in ['size_multiplier','speed_scale','r_r','r_o','r_a']: values[key] = float(variables[key].get())
            agents[name] = values
        names = [variables['name'].get().strip() for variables in self.agent_vars]
        for name in names:
            for key in ('pursues','flees','captures'):
                agents[name][key] = []
        for (i,j), variables in self.relationship_vars.items():
            for key, value in variables.items():
                if value.get():
                    agents[names[i]][key].append(names[j])
        config['agents'] = agents
        resolved,_ = _resolve_settings(settings=config)
        return resolved

    def add_condition(self, path='', values=None):
        row = ttk.Frame(self.condition_body)
        row.pack(fill='x', pady=3)
        path_var = tk.StringVar(value=path)
        values_var = tk.StringVar(value=json.dumps(values) if values is not None else '[]')
        item = (path_var, values_var)
        self.condition_vars.append(item)
        label_var = tk.StringVar()
        selector = ttk.Combobox(row, textvariable=label_var, state='readonly', width=40)
        options = {}
        def refresh():
            options.clear()
            options.update(self.condition_options())
            selector.configure(values=list(options))
            canonical = path_var.get().split('.')
            if len(canonical) == 3 and canonical[0] == 'agents':
                canonical[-1] = {'n':'n_default', 'repulsion':'r_r', 'orientation':'r_o', 'attraction':'r_a'}.get(canonical[-1], canonical[-1])
            label_var.set(next((label for label, value in options.items() if value == '.'.join(canonical)),
                               path_var.get() or 'Choose a parameter'))
        selector.configure(postcommand=refresh)
        selector.bind('<<ComboboxSelected>>', lambda event: path_var.set(options[label_var.get()]))
        refresh()
        selector.pack(side='left', padx=3)
        ttk.Entry(row, textvariable=values_var).pack(side='left', fill='x', expand=True, padx=3)
        def remove():
            self.condition_vars.remove(item)
            row.destroy()
        ttk.Button(row, text='Remove', command=remove).pack(side='left')

    def condition_options(self):
        """Offer scientific fields, including loaded distribution parameters."""
        labels = dict(n_default='Number of agents', speed_scale='Speed multiplier',
                      r_r='Repulsion radius', r_o='Orientation radius', r_a='Attraction radius',
                      max_turn='Maximum turn (radians)', noise='Heading noise (radians)',
                      visual_range_multiplier='Visual range multiplier')
        options = {}
        for variables, template in zip(self.agent_vars, self.agent_templates):
            name = variables['name'].get().strip()
            for key, value in template.items():
                if isinstance(value, (int, float, bool)) and key != 'size':
                    options[f"{name} — {labels.get(key, key.replace('_', ' ').capitalize())}"] = f'agents.{name}.{key}'
            def distribution_fields(values, prefix, title):
                for key, value in values.items():
                    if isinstance(value, dict):
                        distribution_fields(value, f'{prefix}.{key}', f"{title} / {labels.get(key, key.replace('_', ' '))}")
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        options[f'{title} / {key}'] = f'{prefix}.{key}'
            distribution_fields(template.get('distributions', {}), f'agents.{name}.distributions', f'{name} — Distribution')
        for section, fields in (
            ('environment', ['width','height','bounded','spawn_radius','wall_avoid_radius','wall_strength','thigmotaxis']),
            ('movement', ['desired_speed','fov_deg','repulsion_fov_deg','visual_range','capture_distance'])):
            for key in fields:
                # Only offer optional keys that are present in the scientific scenario.
                if key == 'spawn_radius' and key not in self.loaded:
                    continue
                title = {'fov_deg':'Field of view (degrees)', 'repulsion_fov_deg':'Repulsion field of view (degrees)',
                         'desired_speed':'Movement speed', 'thigmotaxis':'Wall following'}.get(key, key.replace('_',' ').capitalize())
                options[f"{section.capitalize()} — {title}"] = f'{section}.{key}'
        options['Movement — Model'] = 'model'
        return options

    def conditions(self, config):
        result = {}
        for path_var, values_var in self.condition_vars:
            path = path_var.get().strip()
            if not path or path in result:
                raise ValueError('Each condition needs a unique parameter path')
            text = values_var.get().strip()
            result[path] = json.loads(text if text.startswith('[') else '[' + text + ']')
        grid = _condition_grid(scenario_from_flat(config), result)
        self.condition_count.set(f"{len(grid)} conditions × {config['n_reps']} repeats")
        return result

    def check_conditions(self):
        try:
            self.conditions(self.configuration())
        except (ValueError, TypeError) as exc:
            messagebox.showerror('Invalid conditions', str(exc), parent=self)

    def document(self):
        config = self.configuration()
        return dict(scenario=scenario_from_flat(config), run={k: config[k] for k in _RUN},
                    conditions=self.conditions(config))

    def load(self):
        filename = filedialog.askopenfilename(parent=self,title='Load simulation settings',filetypes=[('JSON settings','*.json')])
        if not filename: return
        try:
            self.apply_settings(_load_file(filename), Path(filename).name)
        except (ValueError,TypeError,OSError,KeyError) as exc:
            messagebox.showerror('Cannot load settings',str(exc),parent=self)

    def apply_settings(self, settings, source='Provided configuration'):
        resolved, _ = _resolve_settings(settings=settings)
        if len(resolved['agents']) > 3:
            raise ValueError('This setup window supports up to three agent types')
        self.loaded = resolved
        from chimesim.configuration import load_file
        original = load_file(settings) if isinstance(settings, (str, Path)) else settings
        for child in self.condition_body.winfo_children():
            child.destroy()
        self.condition_vars.clear()
        for path, values in original.get('conditions', {}).items():
            self.add_condition(path, values)
        for key, var in self.values.items():
            if key != 'output':
                var.set('' if resolved.get(key) is None else str(resolved.get(key, '')))
        self.bounded.set(resolved['bounded'])
        self.record.set(resolved['save_trajectories'])
        models = {values.get('interaction_model', 'zonal') for values in resolved['agents'].values()}
        self.model.set(next(iter(models)) if len(models) == 1 else 'From settings')
        self.agent_vars, self.agent_templates, self.agent_original_names = [], [], []
        self.relationship_vars = {}
        for name, template in resolved['agents'].items():
            self.add_agent(dict(template, _name=name))
        self.source.set(source)

    def save(self):
        try: config=self.document()
        except (ValueError,TypeError) as exc:
            messagebox.showerror('Invalid settings',str(exc),parent=self);return
        filename=filedialog.asksaveasfilename(parent=self,title='Save settings',defaultextension='.json',filetypes=[('JSON settings','*.json')])
        if filename:
            try: Path(filename).write_text(json.dumps(config,indent=2,allow_nan=False),encoding='utf-8')
            except (ValueError,OSError) as exc: messagebox.showerror('Cannot save settings',str(exc),parent=self)

    def choose_output(self):
        folder=filedialog.askdirectory(parent=self,title='Choose output folder')
        if folder:self.values['output'].set(folder)

    def choose_mask(self):
        filename=filedialog.askopenfilename(parent=self,title='Choose wall mask',filetypes=[('Images','*.png *.jpg *.bmp *.tif'),('All files','*')])
        if filename:self.values['mask'].set(filename)

    def start(self):
        try:
            config=self.configuration()
            conditions=self.conditions(config)
            performance=self.performance.get()
            output=self.values['output'].get().strip()
            if not output:raise ValueError('Choose an output folder')
        except (ValueError,TypeError) as exc:
            messagebox.showerror('Invalid experiment',str(exc),parent=self);return
        self.result=None
        self.cancel_event.clear();self.bar['value']=0;self.status.set('Preparing experiment…')
        self.disabled=[]
        def disable(widget):
            for child in widget.winfo_children():
                if isinstance(child,(ttk.Entry,ttk.Button,ttk.Checkbutton,ttk.Combobox)):
                    self.disabled.append((child,str(child.cget('state'))));child.configure(state='disabled')
                disable(child)
        disable(self)
        self.cancel_button.configure(state='normal')
        self.exit_button.configure(state='normal')
        def work():
            try:
                result=run_experiment(config,output,conditions=conditions,performance=performance,progress=lambda value:self.messages.put(('progress',value)),
                                      should_cancel=self.cancel_event.is_set)
                self.messages.put(('done',result))
            except Exception as exc:self.messages.put(('error',str(exc)))
        # Collect closed Tk sessions here, before a worker can collect their variables.
        gc.collect()
        self.worker=threading.Thread(target=work,name='chimesim-experiment',daemon=True);self.worker.start()

    def cancel(self):
        if self.worker is None:
            return
        self.cancel_event.set();self.status.set('Stopping after the current simulation step…')
        self.cancel_button.configure(state='disabled')

    def poll(self):
        try:
            while True:
                kind,data=self.messages.get_nowait()
                if kind=='progress':
                    fraction=data['completed']/data['total'];self.bar['value']=100*fraction
                    self.status.set((f"Condition {data['condition']+1}/{data['conditions']} — " if 'condition' in data else '') + f"Repeat {data['repeat']}/{data['repeats']} — {data['phase']}, step {data['step']} — {100*fraction:.1f}%")
                    remaining=data['elapsed']*(1-fraction)/fraction if fraction else None
                    self.timing.set(f"Elapsed {data['elapsed']:.0f}s"+(f" · Estimated remaining {remaining:.0f}s" if remaining is not None else ''))
                else:
                    for widget,old_state in self.disabled:widget.configure(state=old_state)
                    self.cancel_button.configure(state='disabled');self.worker=None
                    if kind=='done':
                        self.result=data
                        self.status.set(f"{data['status'].capitalize()} — results: {data['output']}")
                    else:
                        self.status.set('Experiment failed; any written data remain in the output folder.')
                        if not self.closing:messagebox.showerror('Experiment failed',data,parent=self)
                    if self.closing:self.destroy();return
        except queue.Empty:pass
        self.after(100,self.poll)

    def close(self):
        if self.worker is not None:
            if messagebox.askyesno('Close runner','Stop the simulation and exit when its data are saved?',parent=self):
                self.closing=True;self.cancel()
        else:self.destroy()
