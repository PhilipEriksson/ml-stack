# GPU tuning

Optional. The stack works at stock settings; these knobs trade a little throughput for temperature, noise, and power.

## Power limit

```bash
nvidia-smi -q -d POWER | grep -i "power limit"   # current limits
sudo nvidia-smi -pl <watts>                      # set
sudo nvidia-smi -pl <default>                    # reset
```

Lowering the power limit is the single most effective change for a card that is thermally or acoustically constrained. Expect a small throughput loss for a large drop in fan noise and temperature.

**Example — RTX 5090 (575 W TDP, 400 W hardware floor):**

| Target | Use case | Impact |
|---|---|---|
| 575 W (default) | max throughput: training, large batch evals | baseline |
| 450 W | balanced serving | ~5–10% slower generation |
| 400 W | quiet serving, minimal loss | ~7–12% slower |

The floor is hardware-enforced, so `-pl` below it is rejected. Persist the setting by adding the command to a startup script — `nvidia-smi` does not remember it across reboots.

## Clock limit

```bash
nvidia-smi -q -d CLOCK | grep -i graphics   # supported clocks
sudo nvidia-smi -lgc <mhz>                  # cap
sudo nvidia-smi -lgc 0                      # remove cap
```

A clock cap is more predictable than a power cap for steady-state serving, but it also caps short bursts. Use one or the other, not both, unless you are measuring.

## Undervolting (Windows / WSL2 only)

NVIDIA's Linux driver does not expose the voltage/frequency curve, so undervolting is only possible from a Windows host — the GPU keeps the setting regardless of which OS runs the workload.

Typical approach with MSI Afterburner: raise the voltage ceiling, flatten the curve above a chosen point, and set the power slider to the target percentage. The goal is the same clocks at lower voltage, not lower clocks.

On native Linux, use the power limit instead. It achieves most of the thermal benefit with none of the tooling.

## vLLM settings that affect power

| Variable | Power-conscious | Throughput | Effect |
|---|---|---|---|
| `GPU_MEMORY_UTILIZATION` | `0.85` | `0.95` | lower = less VRAM pressure and less compute work |
| `MAX_NUM_SEQS` | `1` | `4`+ | fewer concurrent sequences = less power |
| `PERFORMANCE_MODE` | `interactivity` | `throughput` | interactivity prioritises latency over batch size |

A power-conscious single-user config:

```
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_SEQS=1
PERFORMANCE_MODE=interactivity
```

## Measuring

```bash
nvidia-smi --query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used \
           --format=csv -l 1
```

Change one knob at a time and re-run the same benchmark. Power and clock settings interact, and a change that helps one workload can hurt another.
