from typing import Dict, TextIO

def init_log(f: TextIO, header: Dict[str, float]):
    for k, v in header.items():
        f.write(f"# {k},{v}\n")
    f.write("epoch,mean,median,min,max,std\n")

def log_epoch(f: TextIO, epoch: int, stats: Dict[str, float]):
    f.write(
        f"{epoch},{stats['mean']},{stats['median']},"
        f"{stats['min']},{stats['max']},{stats['std']}\n"
    )

def write_reach_thresholds(f: TextIO, reached: Dict[float, int|None]):
    f.write("# epochs_to_reach\n")
    for t, e in reached.items():
        f.write(f"# reach_{int(t*100)}%,{e}\n")
