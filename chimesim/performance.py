"""Optional workload reporting using standard-library clocks and system metadata."""
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version


def _sysctl(key):
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True,
                                text=True, timeout=2, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def system_info():
    """Best-effort machine details; unavailable fields remain None."""
    memory = None
    try:
        memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if memory <= 0:
            memory = None
    except (AttributeError, OSError, ValueError):
        pass
    if platform.system() == "Darwin" and memory is None:
        value = _sysctl("hw.memsize")
        memory = int(value) if value and value.isdigit() else None
    if platform.system() == "Windows":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total_physical", "available_physical", "total_page", "available_page",
                 "total_virtual", "available_virtual", "available_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory = status.total_physical
        except (AttributeError, OSError):
            pass
    packages = {}
    for name in ("chimesim", "numpy", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(computer_name=platform.node(), operating_system=platform.platform(),
                architecture=platform.machine(), processor=platform.processor() or None,
                logical_cpu_count=os.cpu_count(), total_memory_bytes=memory,
                python_version=platform.python_version(), packages=packages)


def workload(elapsed, cpu, steps, updates, **details):
    """Rates include setup, burn-in, metrics, output and callback costs."""
    return dict(**details, elapsed_seconds=elapsed, process_cpu_seconds=cpu,
                completed_steps=steps, agent_updates=updates,
                steps_per_minute=60 * steps / elapsed if elapsed > 0 else 0.,
                agent_updates_per_second=updates / elapsed if elapsed > 0 else 0.)


def report(conditions, elapsed, cpu, *, verbose):
    result = workload(elapsed, cpu, sum(c['completed_steps'] for c in conditions),
                      sum(c['agent_updates'] for c in conditions),
                      system=system_info(), conditions=conditions,
                      timing_scope="Wall time includes setup, burn-in, metrics, data collection/writing and callbacks; excludes GUI setup and configuration validation.",
                      workload_definition="One step updates the whole population; agent updates count live agents before each step, including burn-in. CPU time covers this process and its threads, not other processes.")
    if verbose:
        print("Performance (including setup, burn-in and output):")
        for condition in conditions:
            print(f"  Condition {condition['condition']}: N={condition['initial_agents']}, "
                  f"{condition['completed_steps']:,} steps in {condition['elapsed_seconds']:.2f}s, "
                  f"{condition['steps_per_minute']:,.0f} steps/min, "
                  f"{condition['agent_updates_per_second']:,.0f} agent updates/s")
    return result
