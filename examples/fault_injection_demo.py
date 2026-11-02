"""
Control-flow demonstration of DynamicBatchController under synthetic OOM events.
"""

from red.config import RedConfig
from red.engines.dbc import DynamicBatchController
from red.state import RedSharedState, RedState


def main() -> None:
    config = RedConfig(min_batch_size=4)
    state = RedSharedState()
    state.state_machine.transition_to(RedState.CONFIGURED)
    state.state_machine.transition_to(RedState.RUNNING)

    dbc = DynamicBatchController(config, state, initial_batch_size=16)
    print(f"[INIT] batch_size={dbc.current_batch_size}, accum={dbc.accumulation_steps}")

    dbc.handle_oom(global_step=1)
    print(f"[OOM 1] batch_size={dbc.current_batch_size}, accum={dbc.accumulation_steps}")

    dbc.handle_oom(global_step=2)
    print(f"[OOM 2] batch_size={dbc.current_batch_size}, accum={dbc.accumulation_steps}")

    try:
        dbc.handle_oom(global_step=3)
    except RuntimeError as exc:
        print(f"[MIN BATCH OOM] Caught expected RuntimeError: {exc}")


if __name__ == "__main__":
    main()