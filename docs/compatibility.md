# Python and operating-system compatibility

For sharing CHIMESim, use the latest **Python 3.12** patch release with working Tkinter as a conservative common setup. Python 3.11–3.14 are the automated compatibility targets; Python 3.15 is an exploratory check until dependencies and GUI tests establish support. This is a testing policy, not a claim that every target has already passed.

## What has been checked

- Existing macOS/Python 3.9 environment: model tests and real Tk GUI checks, including recording/export and reopening.
- Fresh macOS/Python 3.11 and 3.12 environments: model/perception/wall tests pass with newly resolved scientific dependencies. The maintainer additionally confirmed the GUI works well on both versions using python.org installations with Tkinter on an Intel iMac running Sequoia.
- Windows/Linux and Python 3.13/3.14: workflow configured in `.github/workflows/compatibility.yml`; results are pending its first GitHub run.
- Python 3.15: a non-blocking Linux job explores availability and compatibility, without claiming support.

The declared Python minimum stays at 3.9 so existing notebook environments are not broken. The workflow retains a Linux 3.9 regression job. Do not change the minimum or advertise verified cross-platform support until the relevant jobs pass.

## Installation

Start from a checkout of the repository. Keep the existing working environment until its replacement has passed your usual GUI checks.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[masks]"
.venv\Scripts\python -m tkinter
.venv\Scripts\python -c "import chimesim; chimesim.interactive()"
```

macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[masks]'
.venv/bin/python -m tkinter
.venv/bin/python -c 'import chimesim; chimesim.interactive()'
```

On Windows, include Tcl/Tk when installing Python. On macOS, use python.org universal2 installers to obtain matching Python and Tk, particularly on Intel Sequoia where current Homebrew support/binary availability may be limited. Python 3.11.9 and 3.12.10 are older installer-bearing patch releases used for compatibility testing, superseded by newer security patches; do not mistake them for the newest releases. A Homebrew Python requires its matching Tk package. On Linux, install the Tk bindings for the selected interpreter using the distribution package manager (commonly `python3-tk`, or a versioned equivalent). A separate Python interpreter finding `tkinter` does not establish that the notebook kernel's interpreter has it.

Headless simulations need no Tkinter. Pillow is optional for image masks/smooth filled-region display. The notebook extra installs a full Jupyter stack; it is not required to run the GUI. On the current local machine, installation of that full extra in a fresh Python 3.12 environment encountered an unrelated Jupyter dependency requiring unavailable compiler tools; package/model testing uses only masks and matplotlib instead.

For a notebook kernel, install `ipykernel` in the chosen environment and register it with a unique name:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name chimesim-py312 --display-name "CHIMESim (Python 3.12)"
```

Use the chosen environment's Python for these commands.

## Automated and manual checks

The model matrix runs the same numerical tests on Windows, Linux and macOS. Real-GUI jobs open windows, run/pause, draw the existing wall geometry, exercise camera controls, enable asynchronous perception, record/export, save settings and reopen. Linux uses Xvfb. Run locally with:

```bash
python -m pip install -e '.[masks]' matplotlib
python -m unittest discover -s tests -v
```

To include GUI tests, set `CHIMESIM_GUI_TESTS=1` and run the suite with a graphical display. The GUI test is skipped by default, not silently treated as a successful GUI check.

Before sharing, manually check native open/save dialogs, mouse/trackpad scrolling, window resizing, multi-monitor layouts and font/DPI scaling on each platform. Automated window tests cannot guarantee identical native appearance or hardware performance. Seeded trajectories should be compared with numerical tolerances; identical bits across operating systems, scientific-library versions and hardware are not promised.

Sources: [Python Tkinter documentation](https://docs.python.org/3/library/tkinter.html), [SciPy toolchain policy](https://docs.scipy.org/doc/scipy/dev/toolchain.html), [GitHub Python workflows](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).
